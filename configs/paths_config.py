from pathlib import Path

dataset_paths = {
	'celeba_train': Path(''),
	'celeba_test': Path(''),

	'ffhq': Path(''),
	'ffhq_unaligned': Path('')
}

model_paths = {
	# models for backbones and losses
	'ir_se50': Path('pretrained_models/model_ir_se50.pth'),
	# stylegan3 generators
	'stylegan3_ffhq': Path('pretrained_models/stylegan3-r-ffhq-1024x1024.pkl'),
	'stylegan3_ffhq_pt': Path('pretrained_models/sg3-r-ffhq-1024.pt'),
	'stylegan3_ffhq_unaligned': Path('pretrained_models/stylegan3-r-ffhqu-1024x1024.pkl'),
	'stylegan3_ffhq_unaligned_pt': Path('pretrained_models/sg3-r-ffhqu-1024.pt'),
	# model for face alignment
	'shape_predictor': Path('pretrained_models/shape_predictor_68_face_landmarks.dat'),
	# models for ID similarity computation
	'curricular_face': Path('pretrained_models/CurricularFace_Backbone.pth'),
	'mtcnn_pnet': Path('pretrained_models/mtcnn/pnet.npy'),
	'mtcnn_rnet': Path('pretrained_models/mtcnn/rnet.npy'),
	'mtcnn_onet': Path('pretrained_models/mtcnn/onet.npy'),
	# classifiers used for interfacegan training
	'age_estimator': Path('pretrained_models/dex_age_classifier.pth'),
	'pose_estimator': Path('pretrained_models/hopenet_robust_alpha1.pkl')
}

styleclip_directions = {
	"ffhq": {
		'delta_i_c': Path('editing/styleclip_global_directions/sg3-r-ffhq-1024/delta_i_c.npy'),
		's_statistics': Path('editing/styleclip_global_directions/sg3-r-ffhq-1024/s_stats'),
	},
	'templates': Path('editing/styleclip_global_directions/templates.txt')
}

interfacegan_aligned_edit_paths = {
	'age': Path('editing/interfacegan/boundaries/ffhq/age_boundary.npy'),
	'smile': Path('editing/interfacegan/boundaries/ffhq/Smiling_boundary.npy'),
	'pose': Path('editing/interfacegan/boundaries/ffhq/pose_boundary.npy'),
	'Male': Path('editing/interfacegan/boundaries/ffhq/Male_boundary.npy'),
    # Adding interFace GAN+++ boudaries 
    'oval_face': Path('editing/interfacegan/boundaries/ffhq/Oval_Face_boundary.npy'),
	'big_nose': Path('editing/interfacegan/boundaries/ffhq/Big_Nose_boundary.npy'),
	'big_lips': Path('editing/interfacegan/boundaries/ffhq/Big_Lips_boundary.npy'),
    'pointy_nose': Path('editing/interfacegan/boundaries/ffhq/Pointy_Nose_boundary.npy'),
    'attractive' :  Path('editing/interfacegan/boundaries/ffhq/Attractive_boundary.npy'),
    'straight_hair': Path('editing/interfacegan/boundaries/ffhq/Straight_Hair_boundary.npy'),
    'blond_hair': Path('editing/interfacegan/boundaries/ffhq/Blond_Hair_boundary.npy'),
    'black_hair': Path('editing/interfacegan/boundaries/ffhq/Black_Hair_boundary.npy'),
    'brown_hair': Path('editing/interfacegan/boundaries/ffhq/Brown_Hair_boundary.npy'),
    'wear_lipstick': Path('editing/interfacegan/boundaries/ffhq/Wearing_Lipstick_boundary.npy'),
    'narrow_eyes': Path('editing/interfacegan/boundaries/ffhq/Narrow_Eyes_boundary.npy'),
    'mouth_sligtly_open': Path('editing/interfacegan/boundaries/ffhq/Mouth_Slightly_Open_boundary.npy'),
    'pale_skin': Path('editing/interfacegan/boundaries/ffhq/Pale_Skin_boundary.npy'),
    'rossy_cheeks': Path('editing/interfacegan/boundaries/ffhq/Rosy_Cheeks_boundary.npy'),
    'wavy_hair': Path('editing/interfacegan/boundaries/ffhq/Wavy_Hair_boundary.npy'),
    'arched_eyebrows': Path('editing/interfacegan/boundaries/ffhq/Arched_Eyebrows_boundary.npy'),
    'chubby': Path('editing/interfacegan/boundaries/ffhq/Chubby_boundary.npy'),
    'double_chin': Path('editing/interfacegan/boundaries/ffhq/Double_Chin_boundary.npy'),
    'high_cheekbones':Path('editing/interfacegan/boundaries/ffhq/High_Cheekbones_boundary.npy'),
    'bushy_eyebrows': Path('editing/interfacegan/boundaries/ffhq/Bushy_Eyebrows_boundary.npy')
}

interfacegan_unaligned_edit_paths = {
	'age': Path('editing/interfacegan/boundaries/ffhqu/age_boundary.npy'),
	'smile': Path('editing/interfacegan/boundaries/ffhqu/Smiling_boundary.npy'),
	'pose': Path('editing/interfacegan/boundaries/ffhqu/pose_boundary.npy'),
	'Male': Path('editing/interfacegan/boundaries/ffhqu/Male_boundary.npy'),
     # Adding interFace GAN+++ boudaries 
    'oval_face': Path('editing/interfacegan/boundaries/ffhqu/Oval_Face_boundary.npy'),
	'big_nose': Path('editing/interfacegan/boundaries/ffhqu/Big_Nose_boundary.npy'),
	'big_lips': Path('editing/interfacegan/boundaries/ffhqu/Big_Lips_boundary.npy'),
    'pointy_nose': Path('editing/interfacegan/boundaries/ffhqu/Pointy_Nose_boundary.npy'),
    'attractive' :  Path('editing/interfacegan/boundaries/ffhqu/Attractive_boundary.npy'),
    'straight_hair': Path('editing/interfacegan/boundaries/ffhqu/Straight_Hair_boundary.npy'),
    'blond_hair': Path('editing/interfacegan/boundaries/ffhqu/Blond_Hair_boundary.npy'),
    'black_hair': Path('editing/interfacegan/boundaries/ffhqu/Black_Hair_boundary.npy'),
    'brown_hair': Path('editing/interfacegan/boundaries/ffhqu/Brown_Hair_boundary.npy'),
    'wear_lipstick': Path('editing/interfacegan/boundaries/ffhqu/Wearing_Lipstick_boundary.npy'),
    'narrow_eyes': Path('editing/interfacegan/boundaries/ffhqu/Narrow_Eyes_boundary.npy'),
    'mouth_sligtly_open': Path('editing/interfacegan/boundaries/ffhqu/Mouth_Slightly_Open_boundary.npy'),
    'pale_skin': Path('editing/interfacegan/boundaries/ffhqu/Pale_Skin_boundary.npy'),
    'rossy_cheeks': Path('editing/interfacegan/boundaries/ffhqu/Rosy_Cheeks_boundary.npy'),
    'wavy_hair': Path('editing/interfacegan/boundaries/ffhqu/Wavy_Hair_boundary.npy'),
    'arched_eyebrows': Path('editing/interfacegan/boundaries/ffhqu/Arched_Eyebrows_boundary.npy'),
    'chubby': Path('editing/interfacegan/boundaries/ffhqu/Chubby_boundary.npy'),
    'double_chin': Path('editing/interfacegan/boundaries/ffhqu/Double_Chin_boundary.npy'),
    'high_cheekbones':Path('editing/interfacegan/boundaries/ffhqu/High_Cheekbones_boundary.npy'),
    'bushy_eyebrows': Path('editing/interfacegan/boundaries/ffhqu/Bushy_Eyebrows_boundary.npy')
}

