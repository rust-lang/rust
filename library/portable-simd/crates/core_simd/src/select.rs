use crate::simd::{LaneCount, Mask, MaskElement, Simd, SimdElement, SupportedLaneCount};

impl<T, const N: usize> Mask<T, N>
where
    T: MaskElement,
    LaneCount<N>: SupportedLaneCount,
{
    /// Choose elements from two vectors.
    ///
    /// For each element in the mask, choose the corresponding element from `true_values` if
    /// that element mask is true, and `false_values` if that element mask is false.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::{Simd, Mask};
    /// let a = Simd::from_array([0, 1, 2, 3]);
    /// let b = Simd::from_array([4, 5, 6, 7]);
    /// let mask = Mask::from_array([true, false, false, true]);
    /// let c = mask.select(a, b);
    /// assert_eq!(c.to_array(), [0, 5, 6, 3]);
    /// ```
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original inputs"]
    pub fn select<U>(self, true_values: Simd<U, N>, false_values: Simd<U, N>) -> Simd<U, N>
    where
        U: SimdElement<Mask = T>,
    {
        // Safety: The mask has been cast to a vector of integers,
        // and the operands to select between are vectors of the same type and length.
        unsafe { core::intrinsics::simd::simd_select(self.to_int(), true_values, false_values) }
    }

    /// Choose elements from two masks.
    ///
    /// For each element in the mask, choose the corresponding element from `true_values` if
    /// that element mask is true, and `false_values` if that element mask is false.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # use core::simd::Mask;
    /// let a = Mask::<i32, 4>::from_array([true, true, false, false]);
    /// let b = Mask::<i32, 4>::from_array([false, false, true, true]);
    /// let mask = Mask::<i32, 4>::from_array([true, false, false, true]);
    /// let c = mask.select_mask(a, b);
    /// assert_eq!(c.to_array(), [true, false, true, false]);
    /// ```
    #[inline]
    #[must_use = "method returns a new mask and does not mutate the original inputs"]
    pub fn select_mask(self, true_values: Self, false_values: Self) -> Self {
        self & true_values | !self & false_values
    }
}
