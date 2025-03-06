from typing import Optional, Tuple

import numpy as np
import torch

from configs.paths_config import interfacegan_aligned_edit_paths, interfacegan_unaligned_edit_paths
from models.stylegan3.model import GeneratorType
from models.stylegan3.networks_stylegan3 import Generator
from utils.common import tensor2im, generate_random_transform


class FaceEditor:

    def __init__(self, stylegan_generator: Generator, generator_type=GeneratorType.ALIGNED):
        self.generator = stylegan_generator
        if generator_type == GeneratorType.ALIGNED:
            paths = interfacegan_aligned_edit_paths
        else:
            paths = interfacegan_unaligned_edit_paths

        self.interfacegan_directions = {
            'age': torch.from_numpy(np.load(paths['age'])).cuda(),
            'smile': torch.from_numpy(np.load(paths['smile'])).cuda(),
            'pose': torch.from_numpy(np.load(paths['pose'])).cuda(),
            'Male': torch.from_numpy(np.load(paths['Male'])).cuda(),
            'oval_face': torch.from_numpy(np.load(paths['oval_face'])).cuda(),
            'big_nose': torch.from_numpy(np.load(paths['big_nose'])).cuda(),
            'big_lips': torch.from_numpy(np.load(paths['big_lips'])).cuda(),
            "pointy_nose":  torch.from_numpy(np.load(paths['pointy_nose'])).cuda(),
            "attractive":  torch.from_numpy(np.load(paths['attractive'])).cuda(),
            'straight_hair':  torch.from_numpy(np.load(paths['straight_hair'])).cuda(),
            'blond_hair':  torch.from_numpy(np.load(paths['blond_hair'])).cuda(),
            'black_hair': torch.from_numpy(np.load(paths['black_hair'])).cuda(),
            'brown_hair': torch.from_numpy(np.load(paths['brown_hair'])).cuda(),
            'wear_lipstick':  torch.from_numpy(np.load(paths['wear_lipstick'])).cuda(),
            'narrow_eyes':  torch.from_numpy(np.load(paths['narrow_eyes'])).cuda(),
            'mouth_sligtly_open':  torch.from_numpy(np.load(paths['mouth_sligtly_open'])).cuda(),
            'pale_skin':  torch.from_numpy(np.load(paths['pale_skin'])).cuda(),
            'rossy_cheeks':  torch.from_numpy(np.load(paths['rossy_cheeks'])).cuda(),
            'wavy_hair':  torch.from_numpy(np.load(paths['wavy_hair'])).cuda(),
            'arched_eyebrows':  torch.from_numpy(np.load(paths['arched_eyebrows'])).cuda(),
            'chubby': torch.from_numpy(np.load(paths['chubby'])).cuda(),
            'double_chin':  torch.from_numpy(np.load(paths['double_chin'])).cuda(),
            'high_cheekbones':  torch.from_numpy(np.load(paths['high_cheekbones'])).cuda(),
            'bushy_eyebrows':  torch.from_numpy(np.load(paths['bushy_eyebrows'])).cuda(),
            'bag_under_eyes': torch.from_numpy(np.load(paths['bag_under_eyes'])).cuda(),
            'bald':  torch.from_numpy(np.load(paths['bald'])).cuda(),
            'bangs':  torch.from_numpy(np.load(paths['bangs'])).cuda(),
            'blurry':  torch.from_numpy(np.load(paths['blurry'])).cuda(),
            'eyeglasses':  torch.from_numpy(np.load(paths['eyeglasses'])).cuda(),
            'goatee':  torch.from_numpy(np.load(paths['goatee'])).cuda(),
            'gray_hair':  torch.from_numpy(np.load(paths['gray_hair'])).cuda(),
            'heavy_makeup':  torch.from_numpy(np.load(paths['heavy_makeup'])).cuda(),
            'mustache':  torch.from_numpy(np.load(paths['mustache'])).cuda(),
            'no_beard':  torch.from_numpy(np.load(paths['no_beard'])).cuda(),
            'receding_hairline':  torch.from_numpy(np.load(paths['receding_hairline'])).cuda(),
            'sideburns':  torch.from_numpy(np.load(paths['sideburns'])).cuda(),
            'wear_earrings':  torch.from_numpy(np.load(paths['wear_earrings'])).cuda(),
            'wear_hat':  torch.from_numpy(np.load(paths['wear_hat'])).cuda(),
            'wear_necklace':  torch.from_numpy(np.load(paths['wear_necklace'])).cuda(),
            'wear_necktie':  torch.from_numpy(np.load(paths['wear_necktie'])).cuda(),
            'young':  torch.from_numpy(np.load(paths['young'])).cuda(),
        }

    def edit(self, latents: torch.tensor, direction: str, factor: int = 1, factor_range: Optional[Tuple[int, int]] = None,
             user_transforms: Optional[np.ndarray] = None, apply_user_transformations: Optional[bool] = False):
        edit_latents = []
        edit_images = []
        direction = self.interfacegan_directions[direction]
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):
                edit_latent = latents + f * direction
                edit_image, user_transforms = self._latents_to_image(edit_latent,
                                                                     apply_user_transformations,
                                                                     user_transforms)
                edit_latents.append(edit_latent)
                edit_images.append(edit_image)
        else:
            edit_latents = latents + factor * direction
            edit_images, _ = self._latents_to_image(edit_latents, apply_user_transformations)
        return edit_images, edit_latents

    def _latents_to_image(self, all_latents: torch.tensor, apply_user_transformations: bool = False,
                          user_transforms: Optional[torch.tensor] = None):
        with torch.no_grad():
            if apply_user_transformations:
                if user_transforms is None:
                    # if no transform provided, generate a random transformation
                    user_transforms = generate_random_transform(translate=0.3, rotate=25)
                # apply the user-specified transformation
                if type(user_transforms) == np.ndarray:
                    user_transforms = torch.from_numpy(user_transforms)
                self.generator.synthesis.input.transform = user_transforms.cuda().float()
            # generate the images
            images = self.generator.synthesis(all_latents, noise_mode='const')
            images = [tensor2im(image) for image in images]
        return images, user_transforms
