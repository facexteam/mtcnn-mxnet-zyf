# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 06:32:06 2017

@author: zhaoy
"""

import cv2


def cv2_put_text_to_image(img, text, x, y, font_pix_h=10, color=(255, 0, 0)):
    if font_pix_h < 10:
        font_pix_h = 10

    y = y + font_pix_h

    # print img.shape

    h = img.shape[0]

    if x < 0:
        x = 0

    if y > h - 1:
        y = h - font_pix_h

    if y < 0:
        y = font_pix_h

    font_size = font_pix_h / 30.0
    # print font_size
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, color, 1)


def draw_faces(img, bboxes, points=None, draw_score=False):
    if len(bboxes) < 1:
        pass

    for i, bbox in enumerate(bboxes):
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(
            bbox[2]), int(bbox[3])), (0, 255, 0), 1)

        if draw_score and len(bbox) > 4:
            text = '%2.3f' % (bbox[4] * 100)
            cv2_put_text_to_image(img, text, int(bbox[0]), int(bbox[3]), 15)

        if points is not None and points[i] is not None:
            for j in range(5):
                cv2.circle(img, (int(points[i][j]), int(
                    points[i][j + 5])), 2, (0, 0, 255), -1)
