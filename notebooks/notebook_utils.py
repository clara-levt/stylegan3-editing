import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import dlib
import subprocess

from utils.alignment_utils import align_face, crop_face, get_stylegan_transform


ENCODER_PATHS = {
    "restyle_e4e_ffhq": {"id": "1z_cB187QOc6aqVBdLvYvBjoc93-_EuRm", "name": "restyle_e4e_ffhq.pt"},
    "restyle_pSp_ffhq": {"id": "12WZi2a9ORVg-j6d9x4eF-CKpLaURC2W-", "name": "restyle_pSp_ffhq.pt"},
}
INTERFACEGAN_PATHS = {
    "age": {'id': '1NQVOpKX6YZKVbz99sg94HiziLXHMUbFS', 'name': 'age_boundary.npy'},
    "smile": {'id': '1KgfJleIjrKDgdBTN4vAz0XlgSaa9I99R', 'name': 'Smiling_boundary.npy'},
    "pose": {'id': '1nCzCR17uaMFhAjcg6kFyKnCCxAKOCT2d', 'name': 'pose_boundary.npy'},
    "Male": {'id': '18dpXS5j1h54Y3ah5HaUpT03y58Ze2YEY', 'name': 'Male_boundary.npy'},
    # Adding interFace GAN+++ boudaries
    "oval_face": {'id': '102zfQ_2Rv9RVTdukMi3xMuqBTp3i70YK', 'name': 'Oval_Face_boundary.npy'},
    "big_nose": {'id': '1lz_nbsfzUXywzp1v70uGoPLtPwlqg8hA', 'name': 'Big_Nose_boundary.npy'},
    "big_lips": {'id':'10C2DZ3Nsv9n3mAyXEbb1mea-XY06pKcL','name':'Big_Lips_boundary.npy'},
    "pointy_nose": {'id': '1amS402iHbNci9lc9ThsH65a1ateiWwd5','name': 'Pointy_Nose_boundary.npy'},
    "attractive": {'id': '1WL5mc5tDkbP0fD2U2iF1hEfmx7-8K4PH','name':'Attractive_boundary.npy'},
    'straight_hair': {'id':'1BrTPXndDi4wChBJ-bfcPo4Vcs2YE5ELE', 'name':'Straight_Hair_boundary.npy'},
    'blond_hair': {'id':'1iQiJtbptxfnKIRFA5wwfTEZvzIVz4acY','name':'Blond_Hair_boundary.npy'},
    'black_hair':{'id':'1f-fZlpV_x80gc_dVJP1oRDg-AqU_no-N', 'name': 'Black_Hair_boundary.npy'},
    'brown_hair':{'id':'1mrMvRZ_Rzy0iFcLIK8gKmwdoOlygg7-6','name':'Brown_Hair_boundary.npy'},
    'wear_lipstick': {'id':'1zoytQM_kpUVj-BjPX35P8fSWNjzxWZq5','name':'Wearing_Lipstick_boundary.npy'},
    'narrow_eyes': {'id':'12T8ChIIGDbxKeLEkVJG6D6a39SNXqw9s','name':'Narrow_Eyes_boundary.npy'},
    'mouth_sligtly_open': {'id':'1G2-XDjuQvgZQwzxozSzOV2ftlrsMmptr','name':'Mouth_Slightly_Open_boundary.npy'},
    'pale_skin': {'id':'1h-18I4IPqZmRId78YqZ8tnCC1b96Dr7t','name':'Pale_Skin_boundary.npy'},
    'rossy_cheeks': {'id':'1akKTNzbB35RgKUofas-rsTrHiOXC0gx4','name':'Rosy_Cheeks_boundary.npy'},
    'wavy_hair': {'id':'13E8bRbGMooQCRyoThBXtORiBh9eY1Dpn','name': 'Wavy_Hair_boundary.npy'},
    'arched_eyebrows': {'id':'11q9-UjiZuyHGtBavTC_l_LEfpGBdKh70','name':'Arched_Eyebrows_boundary.npy'},
    'chubby': {'id':'14bYK5Z2buYFehFBplKLAT1UKFdA9mNP0','name':'Chubby_boundary.npy'},
    'double_chin': {'id':'1sXoSMgrHiSU8YEBW9-Pgy7bKOyA4RzgG','name':'Double_Chin_boundary.npy'},
    'high_cheekbones': {'id':'1E2UqQyomIB_aGlk99Q8WpRZe9hv2zW6V','name':'High_Cheekbones_boundary.npy'},
    'bushy_eyebrows': {'id':'1kC575Fn_co8zwf9fWgyOi-lyKO6dZwum','name':'Bushy_Eyebrows_boundary.npy'},
}

STYLECLIP_PATHS = {
    "delta_i_c": {"id": "1HOUGvtumLFwjbwOZrTbIloAwBBzs2NBN", "name": "delta_i_c.npy"},
    "s_stats": {"id": "1FVm_Eh7qmlykpnSBN1Iy533e_A2xM78z", "name": "s_stats"},
}


class Downloader:

    def __init__(self, code_dir, use_pydrive, subdir):
        self.use_pydrive = use_pydrive
        current_directory = os.getcwd()
        self.save_dir = os.path.join(os.path.dirname(current_directory), code_dir, subdir)
        os.makedirs(self.save_dir, exist_ok=True)
        if self.use_pydrive:
            self.authenticate()

    def authenticate(self):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)

    def download_file(self, file_id, file_name):
        file_dst = f'{self.save_dir}/{file_name}'
        if os.path.exists(file_dst):
            print(f'{file_name} already exists!')
            return
        if self.use_pydrive:
            downloaded = self.drive.CreateFile({'id': file_id})
            downloaded.FetchMetadata(fetch_all=True)
            downloaded.GetContentFile(file_dst)
        else:
            command = self._get_download_model_command(file_id=file_id, file_name=file_name)
            subprocess.run(command, shell=True, stdout=subprocess.PIPE)

    def _get_download_model_command(self, file_id, file_name):
        """ Get wget download command for downloading the desired model and save to directory ../pretrained_models. """
        url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=self.save_dir)
        return url


def download_dlib_models():
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')


def run_alignment(image_path):
    download_dlib_models()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    print("Aligning image...")
    aligned_image = align_face(filepath=str(image_path), detector=detector, predictor=predictor)
    print(f"Finished aligning image: {image_path}")
    return aligned_image


def crop_image(image_path):
    download_dlib_models()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    print("Cropping image...")
    cropped_image = crop_face(filepath=str(image_path), detector=detector, predictor=predictor)
    print(f"Finished cropping image: {image_path}")
    return cropped_image


def compute_transforms(aligned_path, cropped_path):
    download_dlib_models()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    print("Computing landmarks-based transforms...")
    res = get_stylegan_transform(str(cropped_path), str(aligned_path), detector, predictor)
    print("Done!")
    if res is None:
        print(f"Failed computing transforms on: {cropped_path}")
        return
    else:
        rotation_angle, translation, transform, inverse_transform = res
        return inverse_transform
