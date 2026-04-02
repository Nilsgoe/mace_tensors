import argparse
import logging
import os
from typing import Dict, List, Tuple, Union

import torch
from e3nn import o3

from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.cg import O3_e3nn
from mace.tools.cg_cueq_tools import symmetric_contraction_proj
from mace.tools.scripts_utils import extract_config_mace_model

try:
    import cuequivariance as cue

    CUEQQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUEQQ_AVAILABLE = False
    cue = None

SizeLike = Union[torch.Size, List[int]]


def shapes_match_up_to_unsqueeze(a: SizeLike, b: SizeLike) -> bool:
    if isinstance(a, torch.Tensor):
        a = a.shape
    if isinstance(b, torch.Tensor):
        b = b.shape

    def drop(s):
        return tuple(d for d in s if d != 1)

    return drop(a) == drop(b)


def reshape_like(src: torch.Tensor, ref_shape: torch.Size) -> torch.Tensor:
    try:
        return src.reshape(ref_shape)
    except RuntimeError:
        return src.clone().reshape(ref_shape)


def get_kmax_pairs(
        num_product_irreps: int, correlation: int, num_layers: int, size_mlp: int=0,
) -> List[Tuple[int, int]]:
    """Determine kmax pairs based on num_product_irreps and correlation"""
    if correlation == 2:
        kmax_pairs = [[i, num_product_irreps] for i in range(num_layers - 1)]
        kmax_pairs = kmax_pairs + [[num_layers - 1, size_mlp]]
        return kmax_pairs
    if correlation == 3:
        kmax_pairs = [[i, num_product_irreps] for i in range(num_layers - 1)]
        kmax_pairs = kmax_pairs + [[num_layers - 1, size_mlp]]
        return kmax_pairs
    raise NotImplementedError(f"Correlation {correlation} not supported")


def transfer_symmetric_contractions(
    source_dict: Dict[str, torch.Tensor],
    target_dict: Dict[str, torch.Tensor],
    num_product_irreps: int,
    products: torch.nn.Module,
    correlation: int,
    num_layers: int,
    use_reduced_cg: bool,
    size_mlp: int=0,
):
    """Transfer symmetric contraction weights"""
    kmax_pairs = get_kmax_pairs(num_product_irreps, correlation, num_layers, size_mlp)
    suffixes = ["_max"] + [f".{i}" for i in range(correlation - 1)]
    for i, kmax in kmax_pairs:
        irreps_in = o3.Irreps(
            irrep.ir for irrep in products[i].symmetric_contractions.irreps_in
        )
        irreps_out = o3.Irreps(
            irrep.ir for irrep in products[i].symmetric_contractions.irreps_out
        )
        if use_reduced_cg:
            wm = torch.concatenate(
                [
                    source_dict[
                        f"products.{i}.symmetric_contractions.contractions.{k}.weights{j}"
                    ]
                    for k in range(kmax + 1)
                    for j in suffixes
                ],
                dim=1,
            )
        else:
            wm = torch.concatenate(
                [
                    source_dict[
                        f"products.{i}.symmetric_contractions.contractions.{k}.weights{j}"
                    ]
                    for k in range(kmax + 1)
                    for j in suffixes
                    if not source_dict.get(
                        f"products.{i}.symmetric_contractions.contractions.{k}.weights{j.replace('.', '_')}_zeroed",
                        False,
                    )
                ],
                dim=1,
            )
        if use_reduced_cg:
            _, proj = symmetric_contraction_proj(
                cue.Irreps(O3_e3nn, str(irreps_in)),
                cue.Irreps(O3_e3nn, str(irreps_out)),
                list(range(1, correlation + 1)),
            )
            proj = torch.tensor(proj, dtype=wm.dtype, device=wm.device)
            wm = torch.einsum("zau,ab->zbu", wm, proj)
        target_dict[f"products.{i}.symmetric_contractions.weight"] = wm


def transfer_weights(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    num_product_irreps: int,
    correlation: int,
    num_layers: int,
    use_reduced_cg: bool,
    size_mlp: int=0,
):
    """Transfer weights with proper remapping"""
    # Get source state dict
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    products = source_model.products
    # Transfer symmetric contractions
    transfer_symmetric_contractions(
        source_dict,
        target_dict,
        num_product_irreps,
        products,
        correlation,
        num_layers,
        use_reduced_cg,
        size_mlp,
    )

    transferred_keys = set()
    remaining_keys = (
        set(source_dict.keys()) & set(target_dict.keys()) - transferred_keys
    )
    remaining_keys = {k for k in remaining_keys if "symmetric_contraction" not in k}
    if remaining_keys:
        for key in remaining_keys:
            print("KEY:",key)
            src = source_dict[key]
            tgt = target_dict[key]
            if source_dict[key].shape == target_dict[key].shape:
                logging.debug(f"Transferring additional key: {key}")
                target_dict[key] = source_dict[key]
            elif shapes_match_up_to_unsqueeze(src.shape, tgt.shape):
                logging.debug(
                    f"Transferring key {key} after adapting shape "
                    f"{tuple(src.shape)} → {tuple(tgt.shape)} -> {reshape_like(src, tgt.shape).shape}"
                )
                target_dict[key] = reshape_like(src, tgt.shape)
            else:
                logging.debug(
                    f"Shape mismatch for key {key}: "
                    f"source {source_dict[key].shape} vs target {target_dict[key].shape}"
                )
    # Transfer avg_num_neighbors
    for i in range(num_layers):
        target_model.interactions[i].avg_num_neighbors = source_model.interactions[
            i
        ].avg_num_neighbors

    # Load state dict into target model
    target_model.load_state_dict(target_dict)


def run(
    input_model,
    output_model="_cueq.model",
    device="cpu",
    return_model=True,
):
    # Setup logging

    # Load original model
    # logging.warning(f"Loading model")
    # check if input_model is a path or a model
    if isinstance(input_model, str):
        source_model = torch.load(input_model, map_location=device)
    else:
        source_model = input_model
    default_dtype = next(source_model.parameters()).dtype
    torch.set_default_dtype(default_dtype)
    # Extract configuration
    config = extract_config_mace_model(source_model)

    # Get max_L and correlation from config
    num_product_irreps = len(config["hidden_irreps"].slices()) - 1
    correlation = config["correlation"]
    use_reduced_cg = config.get("use_reduced_cg", True)

    # Add cuequivariance config
    config["cueq_config"] = CuEquivarianceConfig(
        enabled=True,
        layout="ir_mul",
        group="O3_e3nn",
        optimize_all=True,
        conv_fusion=(device == "cuda"),
    )

    # Create new model with cuequivariance config
    logging.info("Creating new model with cuequivariance settings")
    target_model = source_model.__class__(**config).to(device)
    
    size_mlp=len([irreps for irreps in config["MLP_irreps"]]) - 1
    print("RRRR",size_mlp,config["MLP_irreps"])
    print("Source",source_model)
    print("target",target_model)
    torch.set_printoptions(threshold=torch.inf)
    print("\n #################################################### \n")
    #print("SOURCE linear_1 weight:")
    #print(source_model.readouts[-1].linear_1.weight)

    #print("SOURCE linear_2 weight:")
    #print(source_model.readouts[-1].linear_2.weight)
    print("SOURCE equi_nonlin_weight:")
    #print(source_model.readouts[-1].equivariant_nonlin.weight)
    print(source_model.readouts[-1].equivariant_nonlin)
    print(source_model.readouts[-1].equivariant_nonlin.act_scalars)
    print(source_model.readouts[-1].equivariant_nonlin.act_gates)
    # Target
    #print("TARGET linear_1 weight:")
    #print(target_model.readouts[-1].linear_1.weight)

    #print("TARGET linear_2 weight:")
    #print(target_model.readouts[-1].linear_2.weight)
    #print("TARGET equi_nonlin_weight:")
    #print(target_model.readouts[-1].equivariant_nonlin.weight)
    # Transfer weights with proper remapping
    num_layers = config["num_interactions"]
    transfer_weights(
        source_model,
        target_model,
        num_product_irreps,
        correlation,
        num_layers,
        use_reduced_cg,
        size_mlp,
    )
    #print("TARGET linear_1 weight:")
    #print(target_model.readouts[-1].linear_1.weight)

    #print("TARGET linear_2 weight:")
    #print(target_model.readouts[-1].linear_2.weight)
    print("TARGET equi_nonlin_weight:")
    print(target_model.readouts[-1].equivariant_nonlin)
    print(target_model.readouts[-1].equivariant_nonlin.act_scalars)
    print(target_model.readouts[-1].equivariant_nonlin.act_gates)
   
    source_model.eval()
    target_model.eval()
     
    readout_src = source_model.readouts[-1]
    readout_tgt = target_model.readouts[-1]
   
    in_dim = readout_src.linear_1.irreps_in.dim
    torch.manual_seed(0)

    x = torch.randn(1, in_dim, dtype=next(source_model.parameters()).dtype)

    # move to correct device
    x = x.to(next(source_model.parameters()).device)

    z_src = readout_src.linear_1(x)
    z_tgt = readout_tgt.linear_1(x)

    print("linear_1 max |Δ|:", (z_src - z_tgt).abs().max().item())
    
    g_src = readout_src.equivariant_nonlin(z_src)
    g_tgt = readout_tgt.equivariant_nonlin(z_tgt)

    print("Gate max |Δ|:", (g_src - g_tgt).abs().max().item()) 

    y_src = readout_src.linear_2(g_src)
    y_tgt = readout_tgt.linear_2(g_tgt)

    print("Full readout max |Δ|:", (y_src - y_tgt).abs().max().item())

    gate_dim = readout_src.irreps_nonlin.dim
    z = torch.randn(1, gate_dim, dtype=x.dtype, device=x.device)

    g_src = readout_src.equivariant_nonlin(z)
    g_tgt = readout_tgt.equivariant_nonlin(z)

    print("Gate-only max |Δ|:", (g_src - g_tgt).abs().max().item())

    y_src = readout_src.linear_2(g_src)
    y_tgt = readout_tgt.linear_2(g_tgt)

    print("Gate+l2 max |Δ|:", (y_src - y_tgt).abs().max().item())

    #print("Weight comparison:\n",source_model.readouts[-1].linear_1.weight - target_model.readouts[-1].linear_1.weight)
    #print(torch.isclose(target_model.readouts[-1].linear_1.weight,source_model.readouts[-1].linear_1.weight))


    #print(target_model.readouts[-1].equivariant_nonlin.weight)
    if return_model:
        return target_model

    if isinstance(input_model, str):
        base = os.path.splitext(input_model)[0]
        output_model = f"{base}.{output_model}"
    logging.warning(f"Saving CuEq model to {output_model}")
    torch.save(target_model, output_model)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", help="Path to input MACE model")
    parser.add_argument(
        "--output_model",
        help="Path to output cuequivariance model",
        default="cueq_model.pt",
    )
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument(
        "--return_model",
        action="store_false",
        help="Return model instead of saving to file",
    )
    args = parser.parse_args()

    run(
        input_model=args.input_model,
        output_model=args.output_model,
        device=args.device,
        return_model=args.return_model,
    )


if __name__ == "__main__":
    main()
