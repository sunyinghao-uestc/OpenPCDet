# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
# Install from PyPI dependencies (spconv must be installed separately based on CUDA version)
pip install -r requirements.txt

# Build the package with CUDA extensions (iou3d_nms, roiaware_pool3d, roipoint_pool3d, pointnet2, bev_pool, ingroup_inds)
python setup.py develop
```

CUDA extensions live under `pcdet/ops/` with their C++/CUDA sources in `src/` subdirectories. Build failures here are typically CUDA toolkit version mismatches with `torch` or `spconv`.

## Training & Evaluation

All commands run from `tools/`.

**Single GPU training:**
```bash
python train.py --cfg_file ${CONFIG_FILE} [--batch_size B] [--epochs E] [--ckpt CKPT_PATH]
```

**Multi-GPU training:**
```bash
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

**Test a single checkpoint:**
```bash
python test.py --cfg_file ${CONFIG_FILE} --batch_size B --ckpt ${CHECKPOINT_PATH}
```

**Evaluate all checkpoints in an output directory (with live-update polling):**
```bash
python test.py --cfg_file ${CONFIG_FILE} --batch_size B --eval_all
```

**Override config values from CLI:**
```bash
python train.py --cfg_file ${CONFIG_FILE} --set OPTIMIZATION.LR 0.001
```

## Data Preparation

Each dataset requires generating info files before training. Run from repo root:

```bash
# KITTI
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

# NuScenes (lidar-only)
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval

# NuScenes (multi-modal with camera)
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval --with_cam

# Waymo (single-frame)
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml

# Waymo (multi-frame for temporal models like MPPNet)
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos --cfg_file tools/cfgs/dataset_configs/waymo_dataset_multiframe.yaml

# ONCE
python -m pcdet.datasets.once.once_dataset --func create_once_infos --cfg_file tools/cfgs/dataset_configs/once_dataset.yaml

# Argoverse2
python -m pcdet.datasets.argo2.argo2_dataset --root_path data/argo2/sensor --output_dir data/argo2

# Lyft
python -m pcdet.datasets.lyft.lyft_dataset --func create_lyft_infos --cfg_file tools/cfgs/dataset_configs/lyft_dataset.yaml
```

## Architecture

### Data-Model Separation

The codebase enforces a strict separation between datasets and models via a shared coordinate convention. Datasets are responsible for loading raw point clouds and labels, then normalizing them to a unified LiDAR coordinate frame defined by `POINT_CLOUD_RANGE`. Models consume this standardized representation and never see raw dataset formats. This makes all models work with all datasets just by pairing the right config files.

### Config System (`pcdet/config.py`)

Configs are YAML files with hierarchical merge semantics:
- A top-level model config (e.g., `tools/cfgs/kitti_models/pointpillar.yaml`) contains `CLASS_NAMES`, `DATA_CONFIG`, `MODEL`, and `OPTIMIZATION` sections.
- `DATA_CONFIG._BASE_CONFIG_` points to a per-dataset config (e.g., `cfgs/dataset_configs/kitti_dataset.yaml`) that is merged recursively via `merge_new_config()`.
- At runtime, a single global `cfg` EasyDict holds the merged result.
- CLI overrides use dotted keys: `--set MODEL.VFE.NUM_FILTERS [128]`.

### Module Topology (Detector3DTemplate)

All detector models inherit from `Detector3DTemplate` (`pcdet/models/detectors/detector3d_template.py`). Each detector builds itself by progressing through an ordered topology:

```
VFE → BACKBONE_3D → MAP_TO_BEV → PFE → BACKBONE_2D → DENSE_HEAD → POINT_HEAD → ROI_HEAD
```

At each stage, `build_<stage>()` is called. The method returns `(module, model_info_dict)`, where `model_info_dict` threads shared state (voxel sizes, feature channels, grid sizes) across stages. A detector can skip any stage by omitting the corresponding config key — `build_vfe()` returns `None` if `model_cfg.VFE` is missing.

The topology embodies the 3D detection pipeline:
1. **VFE** (Voxel Feature Encoding) — converts raw points into voxel/pillar features
2. **BACKBONE_3D** — sparse 3D convolution backbone (spconv-based)
3. **MAP_TO_BEV** — collapses the height dimension to produce a 2D BEV feature map (e.g., PointPillarScatter, height compression)
4. **PFE** (Point Feature Encoding) — for models that need raw point features alongside voxel features (e.g., PV-RCNN uses voxel set abstraction)
5. **BACKBONE_2D** — 2D CNN backbone on BEV features
6. **DENSE_HEAD** — region proposal network / anchor-based or anchor-free detection head
7. **POINT_HEAD** — additional point-based prediction head
8. **ROI_HEAD** — second-stage refinement for two-stage detectors

### Model Building

`pcdet/models/__init__.py::build_network()` uses the config's `MODEL.NAME` to look up a detector class in `detectors/__all__`. Each detector class (e.g., `PointPillar`, `PVRCNN`, `CenterPoint`) is a thin subclass of `Detector3DTemplate` that implements `forward()` and optionally customizes build methods. The detector's `__init__` calls `self.build_networks()`, which iterates the topology.

### Key Directories

| Directory | Purpose |
|---|---|
| `pcdet/datasets/` | Per-dataset dataset classes, each providing data loading, preprocessing, and evaluation. `dataset.py` has `DatasetTemplate` base class with `collate_batch`. |
| `pcdet/models/detectors/` | Top-level detector classes (PointPillar, SECOND, PV-RCNN, CenterPoint, VoxelRCNN, PartA2, PointRCNN, CaDDN, PVRCNN++, MPPNet, PillarNet, VoxelNeXt, TransFusion, BevFusion). |
| `pcdet/models/backbones_3d/` | Voxel/pillar feature encoders (VFE variants) and sparse 3D conv backbones (spconv, DSVT, Focal SpConv). |
| `pcdet/models/backbones_2d/` | BEV feature map backbones (BaseBEVBackbone) and BEV fusers for multi-modal fusion. |
| `pcdet/models/dense_heads/` | Detection heads: anchor-based (AnchorHeadSingle/Multi), anchor-free (CenterHead, VoxelNeXtHead), point-based (PointHeadSimple/Box). |
| `pcdet/models/roi_heads/` | Second-stage refinement heads for two-stage detectors (PV-RCNN, VoxelRCNN, PartA2, PointRCNN, MPPNet). |
| `pcdet/models/backbones_image/` | Image backbones (Swin Transformer) and necks (GeneralizedLSS) for multi-modal models. |
| `pcdet/models/view_transforms/` | View transformation modules for fusing camera images into BEV space (DepthLSS). |
| `pcdet/ops/` | CUDA/C++ custom ops: `iou3d_nms`, `roiaware_pool3d`, `roipoint_pool3d`, `pointnet2` (ball query, grouping, sampling, interpolation), `bev_pool`. |
| `pcdet/utils/` | Box ops, loss utilities, calibration handling (KITTI), common utilities. |
| `tools/cfgs/` | YAML config files organized by dataset (`kitti_models/`, `waymo_models/`, `nuscenes_models/`, etc.) with shared `dataset_configs/`. |
| `tools/train_utils/` | Training loop (`train_model`), optimizer/scheduler builders. |
| `tools/eval_utils/` | Evaluation harness (`eval_one_epoch`). |
| `tools/visual_utils/` | Open3D-based visualization utilities. |

### Dataset Pipeline

1. `DatasetTemplate.prepare_data()` is the shared preprocessing entry point called by `__getitem__`.
2. During training, `DataAugmentor` applies augmentations (GT sampling, world flip/rotation/scaling) before feature encoding.
3. `PointFeatureEncoder` computes per-point features (absolute coordinates, offsets, etc.).
4. `DataProcessor` runs a sequence of configured processors (mask outside range, shuffle points, voxelization) to produce the model-ready `voxels`, `voxel_coords`, `voxel_num_points`.
5. `collate_batch` assembles variable-sized batch data into dense tensors, adding a batch index to voxel coords and points.

### Distributed Training

Uses `torch.distributed` with NCCL backend. Launcher modes: `pytorch` (via `torch.distributed.launch`), `slurm`. The `cfg.LOCAL_RANK` is set at startup; `cfg.LOCAL_RANK == 0` guards logging and checkpoint saving. `SyncBatchNorm` is available via `--sync_bn`.

### Known Portability Constraints

- **spconv** is a critical dependency but its package name varies by CUDA version (`spconv`, `spconv-cu113`, `spconv-cu117`, etc.). It is intentionally omitted from `setup.py` install_requires.
- The codebase handles two distinct spconv weight layout conventions (1.x vs 2.x) in `_load_state_dict()` — checkpoint weights are transposed to match whichever spconv version is installed.
- Checkpoint compatibility: `version` field in saved checkpoints tracks the OpenPCDet version that produced them.

### Output Structure

Training produces the following under `output/${EXP_GROUP_PATH}/${TAG}/${EXTRA_TAG}/`:
- `ckpt/` — model checkpoints (epoch-based: `checkpoint_epoch_${N}.pth`, plus periodic time-based saves)
- `tensorboard/` — TensorBoard event files
- `log_*.txt` — training logs
- `eval/` — evaluation results

### Custom Datasets

Use `pcdet/datasets/custom/custom_dataset.py` as a template. The key contract: implement `__getitem__` to return a dict with `points`, `gt_boxes`, `gt_names`, and call `self.prepare_data(data_dict)`. The base class handles everything else.
