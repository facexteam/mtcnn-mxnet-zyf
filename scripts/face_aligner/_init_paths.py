import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.abspath('../../'))

# add mxnet path
# add_path('/opt/mxnet')
