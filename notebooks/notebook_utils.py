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
    "oval_face": {'id': '1Q_BuZS42b81hbf2xKO1syI72ZLzk4RYf', 'name': 'Oval_Face_boundary.npy'},
    "big_nose": {'id': '1is7BUj5D6FCSi2iqRLudRyHFQwjgW3g6', 'name': 'Big_Nose_boundary.npy'},
    "big_lips": {'id':'1QCAVpzdsiZo-DV4aBaaTWdG1_2vypepo','name':'Big_Lips_boundary.npy'},
    "pointy_nose": {'id': '12GDUBgfsBpj4Dxw_whEbKOraICOhd6Ew','name': 'Pointy_Nose_boundary.npy'},
    "attractive": {'id': '1gjWcWuAs3oktqiup47a6wJZXdDO30ekc','name':'Attractive_boundary.npy'},
    'straight_hair': {'id':'13y_TuurStYG35cGEldOPioMKyoEIp9uT', 'name':'Straight_Hair_boundary.npy'},
    'blond_hair': {'id':'16lXe-3I-cfPisSu-9sPqJrFRIduIBDPg','name':'Blond_Hair_boundary.npy'},
    'black_hair':{'id':'1RjR-RG_3UU6jF9PSy5v1ioLDD5qUidKp', 'name': 'Black_Hair_boundary.npy'},
    'brown_hair':{'id':'1Kr5t6MT6NW6CCFGC4jfrTSGICi3UAYwF','name':'Brown_Hair_boundary.npy'},
    'wear_lipstick': {'id':'15NfovFTs3RBF5G7IYukR4hIPAdZ1NEee','name':'Wearing_Lipstick_boundary.npy'},
    'narrow_eyes': {'id':'1fuZ6xiCL4v-Pzk9ry_HPV-FcDdrtf2Nj','name':'Narrow_Eyes_boundary.npy'},
    'mouth_sligtly_open': {'id':'1CtCS9EJZzrbXPSxpr4s5gL2UsbwxZC51','name':'Mouth_Slightly_Open_boundary.npy'},
    'pale_skin': {'id':'169W9aYvoxSAxinA3calJ6vvw5F4DJywa','name':'Pale_Skin_boundary.npy'},
    'rossy_cheeks': {'id':'1fac4O3RT9BLgTbeNaogeHZZNpdnGAcon','name':'Rosy_Cheeks_boundary.npy'},
    'wavy_hair': {'id':'1jCNxtfHb1oaR63AqlmRHmUpPfN-UZ1dn','name': 'Wavy_Hair_boundary.npy'},
    'arched_eyebrows': {'id':'1-ib5htTg1f8r9fFa5LuGq7jPpQWhDn2h','name':'Arched_Eyebrows_boundary.npy'},
    'chubby': {'id':'1FwpsDY4iqRjmhBCdaGtsbP4sKjhxueZL','name':'Chubby_boundary.npy'},
    'double_chin': {'id':'1qynI23Hwe_vxa9KCrApldJR-n7JIydqC','name':'Double_Chin_boundary.npy'},
    'high_cheekbones': {'id':'1PEbv_ZJlFmX024M2LUw-CizyYpx0RhNP','name':'High_Cheekbones_boundary.npy'},
    'bushy_eyebrows': {'id':'1xn0PK6cPlGyAewCv3wwjJfKGijratT0m','name':'Bushy_Eyebrows_boundary.npy'},
    'bag_under_eyes': {'id': '17hlOWvtmy4k6vPSW4VHM1H9a91_yWnry', 'name':'Bags_Under_Eyes_boundary.npy'},
    'bald': {'id': '1L7_4oZg3ZpMJUCVm_x5dnwn_CwEok53d', 'name':'Bald_boundary.npy'},
    'bangs': {'id': '1S28ypNw3Jcha2eYi0cvCrFWPamc-CN-A', 'name':'Bangs_boundary.npy'},
    'blurry': {'id': '1fKpf_RIVZDt4TsjkSldmUdNDnNLn2dET', 'name':'Blurry_boundary.npy'},
    'eyeglasses': {'id': '1dlU6784majdxK4SCJHE-bi1R8xtKj8Te', 'name':'Eyeglasses_boundary.npy'},
    'goatee': {'id': '1j8Rid6wzNLCqMHY4NEyhB8h7f4p3sfJ-', 'name':'Goatee_boundary.npy'},
    'gray_hair': {'id': '1gIatWkOWXIlBkL13JWcyV2rwrBhiz3DY', 'name':'Gray_Hair_boundary.npy'},
    'heavy_makeup': {'id': '1QCkrUme6P_u22QDYONxbz7mwCJ04gNUE', 'name':'Heavy_Makeup_boundary.npy'},
    'mustache': {'id': '1-D1nLbpsalE5BJnpndFjFA3EFa4GLCGu', 'name':'Mustache_boundary.npy'},
    'no_beard': {'id': '1vdkKTVa-R0AZJUJb8bEdGB-EvVjzSyld', 'name':'No_Beard_boundary.npy'},
    'receding_hairline': {'id': '1OmxR-W5X2gLCLjF614WE1qqZQVUbf3wi', 'name':'Receding_Hairline_boundary.npy'},
    'sideburns': {'id': '1fW-lIkTpxkB19mLfxLhNg-5If23omkbZ', 'name':'Sideburns_boundary.npy'},
    'wear_earrings': {'id': '1IqTWyBJlHE-vVB8nZR6I4iKs4WjA0ip9', 'name':'Wearing_Earrings_boundary.npy'},
    'wear_hat': {'id': '1A9WdZqUthnqoeW0z3vKNFndPCESn9yhg', 'name':'Wearing_Hat_boundary.npy'},
    'wear_necklace': {'id': '18k6TL57v4pNfvwinrxNLZBPBvILFnbB3', 'name':'Wearing_Necklace_boundary.npy'},
    'wear_necktie': {'id': '1rpy2de4DyWU2nyJ8vUvU6Ql9Pvtx6PsP', 'name':'Wearing_Necktie_boundary.npy'},
    'young': {'id':'1fhWCycvB6k2QWKpQp-glHd4EsGE5Tur8', 'name':'Young_boundary.npy'},
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
