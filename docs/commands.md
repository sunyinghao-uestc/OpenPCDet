# bevfusion
python test.py --cfg_file cfgs/nuscenes_models/bevfusion.yaml  --batch_size 2 --ckpt /home/danc1nc0de/Projects/CompBaselines/OpenPCDet/checkpoints/bevfusion/cbgs_bevfusion.pth
# transfusion-L
python test.py --cfg_file cfgs/nuscenes_models/transfusion_lidar.yaml  --batch_size 8 --ckpt /home/danc1nc0de/Projects/CompBaselines/OpenPCDet/checkpoints/transfusion-L/cbgs_transfusion_lidar.pth

# voxelnext
python test.py --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml  --batch_size 8 --ckpt /home/danc1nc0de/Projects/CompBaselines/OpenPCDet/checkpoints/voxelnext/voxelnext_nuscenes_kernel1.pth