from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from kp_detection import KPDetectionResult

from .match_container import MatchResult

@dataclass
class PairedDetectionResult:
    """
    Container for keypoint matching result, including query and gallery keypoint detection results and match result.

    Attributes:
    ----------
    query_det_result: KPDetectionResult
        The query keypoint detection result.
    gallery_det_result: KPDetectionResult
        The gallery keypoint detection result.
    match_result: MatchResult
        The match result.
    """
    query_det_result: KPDetectionResult
    gallery_det_result: KPDetectionResult
    match_result: MatchResult

    @property
    def query_matched_coordinates(self) -> np.ndarray:
        """
        The query matched coordinates.

        Returns:
        -------
        np.ndarray:
            The query matched coordinates -> shape: (n, 2).
        """
        query_keypoints = self.query_det_result.coordinates
        match_indices = [match.queryIdx for match in self.match_result]
        return query_keypoints[match_indices]
    
    @property
    def gallery_matched_coordinates(self) -> np.ndarray:
        """
        The gallery matched coordinates.

        Returns:
        -------
        np.ndarray:
            The gallery matched coordinates -> shape: (n, 2).
        """
        gallery_keypoints = self.gallery_det_result.coordinates
        match_indices = [match.trainIdx for match in self.match_result]
        return gallery_keypoints[match_indices]

    @property
    def query_descriptors(self) -> np.ndarray:
        """
        The query descriptors.

        Returns:
        -------
        np.ndarray:
            The query descriptors.
        """
        if self.query_det_result.descriptors is None:
            raise ValueError("Query keypoint detection result has no descriptors.")
        return np.array([self.query_det_result.descriptors[match.queryIdx] for match in self.match_result])

    @property
    def gallery_descriptors(self):
        """
        The gallery descriptors.

        Returns:
        -------
        np.ndarray:
            The gallery descriptors.
        """
        if self.gallery_det_result.descriptors is None:
                raise ValueError("Gallery descriptors are not available.")
        return np.array([self.gallery_det_result.descriptors[match.trainIdx] for match in self.match_result])

    def filter_by_ransac(
        self, 
        ransac_th: float = 3.0
        ) -> PairedDetectionResult:
        """
        Filter the matches by RANSAC.

        Parameters:
        ----------
        ransac_th: float
            RANSAC threshold.

        Returns:
        ----------
        PairedDetectionResult: Filtered matches by RANSAC.
        """
        _, ransac_mask = cv2.findHomography(
            srcPoints=self.query_matched_coordinates,
            dstPoints=self.gallery_matched_coordinates,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_th
            ) # _: shape(3, 3) np.ndarray , mask: shape(n, 1) np.ndarray *0: outlier, *1: inlier

        ransac_mask = ransac_mask.flatten().astype(bool) 
        matches = [match for match, is_inlier in zip(self.match_result.matches, ransac_mask) if is_inlier]
        match_result = MatchResult(matches=matches) # type: ignore
        return self.__class__(
            query_det_result=self.query_det_result,
            gallery_det_result=self.gallery_det_result,
            match_result=match_result
            )

    def __str__(self) -> str:
        return f"PairedDetectionResult(query_det_result={self.query_det_result}, \
        gallery_det_result={self.gallery_det_result}, \
        match_result={self.match_result})"
