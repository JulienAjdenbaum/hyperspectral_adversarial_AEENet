import os,pickle,time,pdb,heapq
import matplotlib.pyplot as plt
from utils.decorator import *
import argparse
import scipy.io

try:
    from .utils import allUsefulModule
    from . import data_sampler
    from . import network_initializor
    from .heapManager import  heapInitializer,heap_updater
    from . import imageLabeler
    from . import lbdaAndSigmaComputer
except:
    from utils import allUsefulModule
    import data_sampler
    from heapManager import   heapInitializer,heap_updater
    from network_initialization import network_edge_initializor
    import imageLabeler,lbdaAndSigmaComputer
import utils.utils

import numpy as np
import skimage.measure
from  networkx.algorithms.traversal.edgebfs import edge_bfs

NetworkEdgeInitializor = network_edge_initializor.NetworkEdgeInitializor
dirFile = os.path.dirname(__file__)
#TODO: use the time compiler on the super pixel algorithm, from the book on efficient pytohn programming
#TODO : factorize the edge value computation
#TODO : add the unit test
#TODO : add the criterion values


from enum import Enum
class IterationMode(Enum):
    UntilConvergence = 1

    #TODO : get back here to continue

class Main:
    """
        implementation of the ERS (entropy rate superpixel algorithm)
        From an image, we seek to construct a superpixel segmentation
        two graph are constructed during the process, and first graph (the initial graph) where each pixel
        is linked to its neighbor, and a second graph (the ouput graph), where only the pixels belonging to each
        other are linked.

    """

    path = os.path.join(dirFile,"pickeld_data/pickled_main.pkl") #file where a serialized copy of the instance is saved
    path = os.path.abspath(path)
    # @profile
    def __init__(self,img,K,sigma):
        self.img = img
        self.shape_img = self.img.shape[:2]
        self.nbNodes = np.product(self.img.shape[:2])
        self.K = K

        lbdaAndSigmaComputer.SigmaReader(sigma)
        graph_init = NetworkEdgeInitializor(self.img)
        self.edges_with_nodes = graph_init.edges_with_nodes
        self.G = graph_init.G
        self.G.add_weighted_edges_from(self.edges_with_nodes,connected = False)

        self.edges_with_nodes = np.array(list(self.G.edges.data()))
        self.heapMin = self.init_heap()

        self.lbda_computer = self.init_lbda_computer()

        self.heap_updater = heap_updater.HeapUpdater(self.heapMin,nbNodes = self.nbNodes,K = self.K)
        self.edges_linked = self.update_heap()


        self.list_linked_nodes = self.get_linked_nodes()

        self.imageLabeler = imageLabeler.ImageLabeler(self.img,self.list_linked_nodes)

    def init_lbda_computer(self):
        alg = lbdaAndSigmaComputer.LbdaComputer(gainH=self.heap_initializer.gainH, gainB=self.heap_initializer.gainB, K=self.K)
        return alg
    def get_linked_nodes(self):
        s1 = [id(el.linked_list_of_nodes) for el in self.G.nodes]
        a, b = np.unique(s1, return_index=True)
        s2 = np.array([el.linked_list_of_nodes for el in self.G.nodes])
        s3 = s2[b]
        return s3
    @timeit()
    def update_heap(self,nbIteration = None):
        if nbIteration is None:
            self.heap_updater.iterate_until_end()
        else:
            self.heap_updater.iterate_multiple(nbIteration)
        return self.heap_updater.edges
    @timeit()
    def init_heap(self):
        """ returns the heap val"""
        self.heap_initializer = heapInitializer.HeapInitializer(self.edges_with_nodes,K = self.K)
        return self.heap_initializer.heapMin


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", "--image",help="get the path to the image, for superpixel segmentation")
    parser.add_argument("-data", "--data_location", help="get the path to the image, for superpixel segmentation")
    parser.add_argument("-sp","--super_pixels", type=int, help="set the numer of superpixels to use for the segmentation",default=20)
    parser.add_argument("-R", "--r", type=int,
                        help="set the numer of superpixels to use for the segmentation", default=20)
    parser.add_argument("-G","--g", type=int, help="set the numer of superpixels to use for the segmentation",default=20)
    parser.add_argument("-B", "--b", type=int,
                        help="set the numer of superpixels to use for the segmentation", default=20)



    args = parser.parse_args()
    path_img = args.image
    data_location = args.data_location
    sp = args.super_pixels
    r = args.r
    g = args.g
    b = args.b

    assert os.path.exists(path_img)
    assert isinstance(sp,int)

    deltas = []
    for i in range(1):
        start = time.time()
        # index = 134
        # path_img,path_seg = data_sampler.get_path_img_and_seg_from_id(index)
        # img = plt.imread(path_img)#[:100,:100]
        #img = plt.imread(path_img)
        mat = scipy.io.loadmat(path_img)
        img = mat[data_location].reshape(mat[data_location].shape[0], int(np.sqrt(mat[data_location].shape[1])), int(np.sqrt(mat[data_location].shape[1])))
        img = np.stack((img[r],img[g],img[b]), axis=2)
        img = img.astype(np.float32)/np.max(img)
        alg = Main(img,K = sp, sigma= 5/255.0)
        stop = time.time()
        delta = stop-start
        deltas.append(delta)
    # print(deltas)
    labelling = alg.imageLabeler.show_image_with_res()
    reshaped_labelling_array = labelling.reshape(-1, labelling.shape[-1])
    unique_colors = np.unique(reshaped_labelling_array, axis=0)
    labelling_to_save = np.zeros(reshaped_labelling_array.shape[0])
    for i in range(len(unique_colors)):
        # print(np.where(reshaped_labelling_array==unique_colors[i])[0].shape)
        # print(np.where(reshaped_labelling_array==unique_colors[i])[0])
        labelling_to_save[np.where(reshaped_labelling_array==unique_colors[i])[0]] = i
    # print(labelling_to_save.shape)
    # print(labelling.shape)
    labelling_to_save = labelling_to_save.reshape(labelling.shape[:2])
    plt.imshow(labelling_to_save)
    plt.show()
    # print(f"{path_img[:-4]}_{sp}_seg.npy")
    np.save(f"{path_img[:-4]}_{sp}_seg.npy", labelling_to_save)

