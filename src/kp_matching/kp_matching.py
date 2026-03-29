import numpy as np

from .match_container import MatchResult
from .parameter import KPMatchingParameters

class KPMatchingProcessor:
    """Class for keypoint matching.

    This class implements various methods for keypoint matching, including brute-force,
    FLANN-based, and k-Nearest Neighbors (kNN) matchers. It supports applying Lowe's 
    ratio test to filter good matches based on a threshold.

    Attributes:
    ----------
    params: KPMatchingParameters
        The parameters for the keypoint matching processor.
    function: Callable[[np.ndarray, np.ndarray], list[cv2.DMatch]]
        The function for matching descriptors.
    """
    def __init__(
        self, 
        params: KPMatchingParameters
        ) -> None:
        """
        Initialize the KPMatchingProcessor.

        Parameters:
        ----------
        params: KPMatchingParameters
            The parameters for the keypoint matching processor.
        """
        self.params = params
        self.function = self.params.define_matching_function()

    def match(
        self, 
        desc1: np.ndarray, 
        desc2: np.ndarray
        ) -> MatchResult: 
        """
        Performs keypoint matching based on the specified method.
        This method matches descriptors from two sets using the chosen matching technique.
        
        Parameters:
        ----------
        desc1: np.ndarray
            Descriptors of keypoints in the first image.
        desc2: np.ndarray
            Descriptors of keypoints in the second image.

        Returns:
        ---------
        MatchResult: A sorted list of matches based on their distance.
        
        Raises:
        --------
        ValueError: If an unsupported matching method is specified.
        """

        matches = MatchResult(matches=self.function(desc1, desc2)) # type: ignore

        if self.params.is_ratio_test_enabled:
            matches = matches.apply_ratio_test(threshold=self.params.ratio_test_threshold)
        return matches

