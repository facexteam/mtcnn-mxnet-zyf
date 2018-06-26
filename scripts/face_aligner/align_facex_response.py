# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 06:33:36 2017

@author: zhaoy
"""
import sys
import os
import os.path as osp

import json
import time
import cv2

import _init_paths
from face_aligner import FaceAligner


def main(json_file, save_dir=None, save_img=True, show_img=True):
    if not osp.exists(json_file):
        print 'Cannot find json file: ' + json_file
        pass

    if save_dir is None:
        save_dir = './fa_facex_rlt'

    save_json = 'mtcnn_align_rlt.json'
    model_path = "../../model"

    fp_json = open(json_file, 'r')
    facex_response = json.load(fp_json)
    fp_json.close()

    if (not facex_response
            or not isinstance(facex_response, dict)
            or 'facex_det' not in facex_response
            ):
        print 'Invalid json file: ' + json_file
        pass

    facex_det_response = facex_response['facex_det']

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp_rlt = open(osp.join(save_dir, save_json), 'w')
    results = []

    for item in facex_det_response:
        img_path = item['name']
        print '===> Processing image: ' + img_path

        if 'detections' not in item:
            continue

        face_rects = []
        for face in item['detections']:
            face_rects.append(face['pts'])

        img = cv2.imread(img_path)

        aligner = FaceAligner(model_path, False)

        rlt = {}
        rlt["filename"] = img_path
        rlt["faces"] = []
        rlt['face_count'] = 0

        t1 = time.clock()
        bboxes, points = aligner.align_face(img, face_rects)
        t2 = time.clock()

        n_boxes = len(face_rects)
        print("-->Alignment cost %f seconds, processed %d face rects, avg time: %f seconds" %
              ((t2 - t1), n_boxes, (t2 - t1) / n_boxes))

        if bboxes is not None and len(bboxes) > 0:
            for (box, pts) in zip(bboxes, points):
                #                box = box.tolist()
                #                pts = pts.tolist()
                tmp = {'rect': box[0:4],
                       'score': box[4],
                       'pts': pts
                       }
                rlt['faces'].append(tmp)

        rlt['face_count'] = len(bboxes)

        rlt['message'] = 'success'
        results.append(rlt)

        spl = osp.split(img_path)
        sub_dir = osp.split(spl[0])[1]
        base_name = spl[1]

        save_img_subdir = osp.join(save_dir, sub_dir)
        if not osp.exists(save_img_subdir):
            os.mkdir(save_img_subdir)

#        save_rect_subdir = osp.join(save_dir, sub_dir)
#        if not osp.exists(save_rect_subdir):
#            os.mkdir(save_rect_subdir)
        # print pts

        save_img_fn = osp.join(save_img_subdir, base_name)
        print 'save face chip into ', save_img_fn

        # facial5points = np.reshape(pts, (2, -1))
        # dst_img = warp_and_crop_face(
        #     img, facial5points, reference_5pts, output_size)
        dst_img = aligner.get_face_chips(img, [box], [pts], True)[0]
        cv2.imwrite(save_img_fn, dst_img)

    json.dump(results, fp_rlt, indent=2)
    fp_rlt.close()

if __name__ == '__main__':
#    json_file = './facex_det_response.json'
    json_file = r'C:\zyf\github\misc-zyf\facex_det_wlc_cvt.json'
    save_dir = None

    main(json_file, save_dir, save_img=True, show_img=False)
