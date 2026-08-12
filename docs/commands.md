conda activate openpcdet

export PYTHONPATH="/home/danc1nc0de/Projects/CompBaselines/OpenPCDet:$PYTHONPATH"

# bevfusion
python test.py --cfg_file cfgs/nuscenes_models/bevfusion.yaml  --batch_size 2 --ckpt /home/danc1nc0de/Projects/CompBaselines/OpenPCDet/checkpoints/bevfusion/cbgs_bevfusion.pth

# transfusion-L
python test.py --cfg_file cfgs/nuscenes_models/transfusion_lidar.yaml  --batch_size 8 --ckpt /home/danc1nc0de/Projects/CompBaselines/OpenPCDet/checkpoints/transfusion-L/cbgs_transfusion_lidar.pth

# voxelnext
python test.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml  --batch_size 8 --ckpt /home/danc1nc0de/Projects/CompBaselines/OpenPCDet/checkpoints/voxelnext/voxelnext_nuscenes_kernel1.pth

# centerpoint
python test.py --cfg_file cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml  --batch_size 8 --ckpt /home/danc1nc0de/Projects/CompBaselines/OpenPCDet/checkpoints/centerpoint_pointpillars/cbgs_pp_centerpoint_nds6070.pth