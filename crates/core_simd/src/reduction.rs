use crate::simd::intrinsics::{
    simd_reduce_add_ordered, simd_reduce_max, simd_reduce_min, simd_reduce_mul_ordered,
};
use crate::simd::{LaneCount, Simd, SupportedLaneCount};

macro_rules! impl_float_reductions {
    { $scalar:ty } => {
        impl<const LANES: usize> Simd<$scalar, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {

            /// Reducing add.  Returns the sum of the lanes of the vector.
            ///
            /// # Examples
            ///
            /// ```
            /// # #![feature(portable_simd)]
            /// # use core::simd::Simd;
            #[doc = concat!("# use core::simd::", stringify!($scalar), "x2;")]
            #[doc = concat!("let v = ", stringify!($scalar), "x2::from_array([1., 2.]);")]
            /// assert_eq!(v.reduce_sum(), 3.);
            /// ```
            #[inline]
            pub fn reduce_sum(self) -> $scalar {
                // LLVM sum is inaccurate on i586
                if cfg!(all(target_arch = "x86", not(target_feature = "sse2"))) {
                    self.as_array().iter().sum()
                } else {
                    // Safety: `self` is a float vector
                    unsafe { simd_reduce_add_ordered(self, 0.) }
                }
            }

            /// Reducing multiply.  Returns the product of the lanes of the vector.
            ///
            /// # Examples
            ///
            /// ```
            /// # #![feature(portable_simd)]
            /// # use core::simd::Simd;
            #[doc = concat!("# use core::simd::", stringify!($scalar), "x2;")]
            #[doc = concat!("let v = ", stringify!($scalar), "x2::from_array([3., 4.]);")]
            /// assert_eq!(v.reduce_product(), 12.);
            /// ```
            #[inline]
            pub fn reduce_product(self) -> $scalar {
                // LLVM product is inaccurate on i586
                if cfg!(all(target_arch = "x86", not(target_feature = "sse2"))) {
                    self.as_array().iter().product()
                } else {
                    // Safety: `self` is a float vector
                    unsafe { simd_reduce_mul_ordered(self, 1.) }
                }
            }

            /// Reducing maximum.  Returns the maximum lane in the vector.
            ///
            /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
            /// return either.
            ///
            /// This function will not return `NaN` unless all lanes are `NaN`.
            ///
            /// # Examples
            ///
            /// ```
            /// # #![feature(portable_simd)]
            /// # use core::simd::Simd;
            #[doc = concat!("# use core::simd::", stringify!($scalar), "x2;")]
            #[doc = concat!("let v = ", stringify!($scalar), "x2::from_array([1., 2.]);")]
            /// assert_eq!(v.reduce_max(), 2.);
            ///
            /// // NaN values are skipped...
            #[doc = concat!("let v = ", stringify!($scalar), "x2::from_array([1., ", stringify!($scalar), "::NAN]);")]
            /// assert_eq!(v.reduce_max(), 1.);
            ///
            /// // ...unless all values are NaN
            #[doc = concat!("let v = ", stringify!($scalar), "x2::from_array([",
                stringify!($scalar), "::NAN, ",
                stringify!($scalar), "::NAN]);"
            )]
            /// assert!(v.reduce_max().is_nan());
            /// ```
            #[inline]
            pub fn reduce_max(self) -> $scalar {
                // Safety: `self` is a float vector
                unsafe { simd_reduce_max(self) }
            }

            /// Reducing minimum.  Returns the minimum lane in the vector.
            ///
            /// Returns values based on equality, so a vector containing both `0.` and `-0.` may
            /// return either.
            ///
            /// This function will not return `NaN` unless all lanes are `NaN`.
            ///
            /// # Examples
            ///
            /// ```
            /// # #![feature(portable_simd)]
            /// # use core::simd::Simd;
            #[doc = concat!("# use core::simd::", stringify!($scalar), "x2;")]
            #[doc = concat!("let v = ", stringify!($scalar), "x2::from_array([3., 7.]);")]
            /// assert_eq!(v.reduce_min(), 3.);
            ///
            /// // NaN values are skipped...
            #[doc = concat!("let v = ", stringify!($scalar), "x2::from_array([1., ", stringify!($scalar), "::NAN]);")]
            /// assert_eq!(v.reduce_min(), 1.);
            ///
            /// // ...unless all values are NaN
            #[doc = concat!("let v = ", stringify!($scalar), "x2::from_array([",
                stringify!($scalar), "::NAN, ",
                stringify!($scalar), "::NAN]);"
            )]
            /// assert!(v.reduce_min().is_nan());
            /// ```
            #[inline]
            pub fn reduce_min(self) -> $scalar {
                // Safety: `self` is a float vector
                unsafe { simd_reduce_min(self) }
            }
        }
    }
}

impl_float_reductions! { f32 }
impl_float_reductions! { f64 }
