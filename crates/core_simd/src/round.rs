use crate::simd::intrinsics;
use crate::simd::{LaneCount, Simd, SupportedLaneCount};

macro_rules! implement {
    {
        $type:ty, $int_type:ty
    } => {
        impl<const LANES: usize> Simd<$type, LANES>
        where
            LaneCount<LANES>: SupportedLaneCount,
        {
            /// Rounds toward zero and converts to the same-width integer type, assuming that
            /// the value is finite and fits in that type.
            ///
            /// # Safety
            /// The value must:
            ///
            /// * Not be NaN
            /// * Not be infinite
            /// * Be representable in the return type, after truncating off its fractional part
            #[inline]
            pub unsafe fn to_int_unchecked(self) -> Simd<$int_type, LANES> {
                unsafe { intrinsics::simd_cast(self) }
            }
        }
    }
}

implement! { f32, i32 }
implement! { f64, i64 }
