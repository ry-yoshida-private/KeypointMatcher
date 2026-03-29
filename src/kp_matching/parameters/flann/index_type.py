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
    FLANN_INDEX_LINEAR              = 0
    FLANN_INDEX_KDTREE              = 1
    FLANN_INDEX_KMEANS              = 2
    FLANN_INDEX_COMPOSITE           = 3
    FLANN_INDEX_KDTREE_SINGLE       = 4
    FLANN_INDEX_HIERARCHICAL        = 5
    FLANN_INDEX_LSH                 = 6 # For binary descriptors
    FLANN_INDEX_SAVED               = 7
    FLANN_INDEX_AUTOTUNED           = 8
