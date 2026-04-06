from enum import IntEnum

class FLANNIndexType(IntEnum):
    """
    FLANN index type.

    Attributes:
    ----------
    FLANN_INDEX_LINEAR: int
        The linear index type.
    FLANN_INDEX_KDTREE: int
        The KD-tree index type.
    FLANN_INDEX_KMEANS: int
        The K-means index type.
    FLANN_INDEX_COMPOSITE: int
        The composite index type.
    FLANN_INDEX_KDTREE_SINGLE: int
        The single KD-tree index type.
    FLANN_INDEX_HIERARCHICAL: int
        The hierarchical index type.
    FLANN_INDEX_LSH: int
        The LSH index type.
    FLANN_INDEX_SAVED: int
        The saved index type.
    FLANN_INDEX_AUTOTUNED: int
        The autotuned index type.
    """
    LINEAR              = 0
    KDTREE              = 1
    KMEANS              = 2
    COMPOSITE           = 3
    KDTREE_SINGLE       = 4
    HIERARCHICAL        = 5
    LSH                 = 6 # For binary descriptors
    SAVED               = 7
    AUTOTUNED           = 8
