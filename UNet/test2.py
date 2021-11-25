# Importing Libraries
import os
import shutil
import pathlib
import pandas as pd
from PIL import Image
# import pathlib as path
import numpy as np


# Setting Paths
dir_data = './accida_segmentation_dataset_v1'
dir_data_new = './dataset_new'
spacing_except_data ={
    'spacing': [
         '20190818_493031566140159775.jpeg',    # 손잡이부분을 이격으로 라벨링함
         '20190818_492061566101029326.jpeg',    # 이격있다고 판단 안됨
         '20190816_489221565923816243.jpeg',    # 트렁크 사이를 이격으로 라벨링함
         '20190812_485061565574086740.jpeg',    # 이격있다고 판단 안됨
         '20190808_480381565257406038.jpeg',    # 이격있다고 판단 안됨
         '20190807_479491565176758472.jpeg',    # 사람팔사진
         '20190805_476671564979440911.jpeg',    # 이미지내 글씨있음
         '20190805_476671564979438562.jpeg',    # 이미지내 글씨있음
         '20190805_476671564979437397.jpeg',    # 이미지내 글씨있음
         '20190805_476671564979436008.jpeg',    # 이미지내 글씨있음
         '20190805_476671564979433113.jpeg',    # 이미지내 글씨있음
         '20190804_475591564899663484.jpeg',    # 차량이미지라고 보기힘듬
         '20190804_475591564899662383.jpeg',    # 차량내부에서 밖을 찍음
         '20190731_470331564555126804.jpeg',    # 차량이미지 아님(영수증)
         '20190731_470331564555124879.jpeg',    # 차량이미지 아님(영수증)
         '20190727_465621564202183181.jpeg',    # 이격이 어마어마한대 마스크가 커버 못침
         '20190723_462041563890240798.jpeg',    # 이미지에 노란색 동그라미
         '20190723_462041563890240389.jpeg',    # 이미지에 노란색 동그라미
         '20190722_460641563784863099.jpeg',    # 이격있지만 마스크 처리 안됨
         '20190717_455021563337341248.jpeg',    # 이미지내 글씨있음
         '20190717_455021563337340508.jpeg',    # 이미지내 글씨있음
         '20190717_455011563337229116.jpeg',    # 이미지내 글씨잇음
         '20190717_455011563337228528.jpeg',    # 이미지내 글씨있음
         '20190717_455011563337227919.jpeg',    # 이미지내 글씨있음
         '20190717_455011563337227221.jpeg',    # 이미지내 글씨있음
         '20190717_455011563337226280.jpeg',    # 이미지내 글씨있음
         '20190715_453281563189200301.jpeg',    # 이미지내 노란색 표시
         '20190715_453281563189199824.jpeg',    # 이미지내 노란색 표시
         '20190715_453281563189199179.jpeg',    # 이미지내 노란색 표시
         '20190624_429791561334936827.jpeg',    # 이미지내 글씨있음
         '20190624_429791561334936271.jpeg',    # 이미지내 글씨있음
         '20190624_429791561334935608.jpeg',    # 이미지내 글씨있음
         '20190624_429791561334936827.jpeg',    # 이미지내 텍스트
         '20190517_393311558019492109.jpeg',    # 이미지 너무 어두움
         '20190507_385581557220678079.jpeg',    # 나무가 메인;
         '20190419_368741555663340993.jpeg',    # 인물사진이 메인;
         '20190227_328701551250213107.jpeg',    # 차량사진인지 구분안감
         '20190225_327221551082752663.jpeg',    # 차량사진아님
         '20190225_327071551072569833.jpeg',    # 오토바이사진
         '20190225_327071551072569244.jpeg',    # 오토바이사진
         '20190225_327071551072566092.jpeg',    # 오토바이사진
         '20190225_327071551072565363.jpeg',    # 오토바이사진
         '20190225_327071551072564602.jpeg',    # 오토바이사진
         '20190218_322321550469489922.jpeg',    # 파일형식 이상
         '20190218_322321550469487907.jpeg',    # 파일형식 이상
         '20190105_291811546668889646.jpeg',    # 오토바이 사진
         '20190105_291811546668889263.jpeg',    # 오토바이 사진
         '20181227_285001545884663968.jpeg',    # 백미러만 따로 있는 사진;
         '20181227_284831545844349746.jpeg'	    # 전봇대가 메인
    ]
}


def del_dataset(root_path, new_data_path, del_dict):
    category_damages = ['dent', 'scratch', 'spacing']
    category_data = ['test', 'train', 'valid']
    category_represent = ['images', 'masks']

    # Making Directories
    for damage in category_damages:
        for data in category_data:
            for represent in category_represent:
                if not os.path.exists(os.path.join(new_data_path, damage, data, represent)):
                    os.makedirs(os.path.join(new_data_path, damage, data, represent))

    # Saving Data
    for damage in category_damages:
        for data in category_data:
            for represent in category_represent:
                cnt = 0
            for file_name in os.listdir(os.path.join(root_path, damage, data, represent)):
                if 'augmented' in file_name:
                    continue
                isin = False

                for key, val in del_dict.items():
                    if (damage == key) and (file_name in val):
                        isin = True
                if isin: continue

                shutil.copyfile(os.path.join(root_path, damage, data, represent, file_name),
                         os.path.join(new_data_path, damage, data, represent, file_name))
                cnt += 1
            print('{0}_{1}_{2}: {3}'.format(damage, data, represent, cnt))
    print("copy done!")


def socar_dataset_df(root_path, damage_type):

    """
    name : 파일명,
    train/valid/mask : train=1, valid=2, mask=3,
    is_mask : mask image에 mask 있으면 1, 없으면 0,
    width : image의 넓이,
    height : image의 높이,
    image_path : 이미지의 경로,
    mask_path : 마스크 이미지 경로,
    """
    df = pd.DataFrame()
    data_type = ["train", "valid", "test"]

    for idx, dt in enumerate(data_type):
        path = pathlib.Path(root_path, damage_type, dt)

        image_path =list(sorted(path.glob('images/*.jpg')))
        mask_path =list(sorted(path.glob('masks/*.jpg')))

        image_path_str = [str(i) for i in image_path]
        mask_path_str = [str(i) for i in mask_path]

        image_name = [i.split('/')[-1].split('.')[0] for i in image_path_str]

        width, height, is_mask = [], [], []

        for img in image_path_str:
            # image size 출력
            image = Image.open(img)
            width.append(image.width)
            height.append(image.height)

            image_gray = image.convert('L')
            img_arr = np.array(image_gray)
            if len(np.where(img_arr > 200)[1]) == 0:
                is_mask.append(0)
            else:
                is_mask.append(1)

        temp_df = pd.DataFrame({'name': image_name, 'train/valid/mask': idx+1, 'is_mask': is_mask,
                                'width': width, 'height': height, 'image_path': image_path_str,
                                'mask_path': mask_path_str})
        df = pd.concat([df, temp_df])

    return df


del_dataset(dir_data, dir_data_new, spacing_except_data)
socar_df = socar_dataset_df(dir_data_new, "dent")
socar_df.to_csv('socar_df.csv')