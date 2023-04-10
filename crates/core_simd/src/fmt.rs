use crate::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
use core::fmt;

impl<T, const LANES: usize> fmt::Debug for Simd<T, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    T: SimdElement + fmt::Debug,
{
    /// A `Simd<T, N>` has a debug format like the one for `[T]`:
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd::Simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd::Simd;
    /// let floats = Simd::<f32, 4>::splat(-1.0);
    /// assert_eq!(format!("{:?}", [-1.0; 4]), format!("{:?}", floats));
    /// ```
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <[T] as fmt::Debug>::fmt(self.as_array(), f)
    }
}
