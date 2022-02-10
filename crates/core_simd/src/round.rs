use crate::simd::intrinsics;
use crate::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
use core::convert::FloatToInt;

macro_rules! implement {
    {
        $type:ty
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
            ///
            /// If these requirements are infeasible or costly, consider using the safe function [cast],
            /// which saturates on conversion.
            ///
            /// [cast]: Simd::cast
            #[inline]
            pub unsafe fn to_int_unchecked<I>(self) -> Simd<I, LANES>
            where
                $type: FloatToInt<I>,
                I: SimdElement,
            {
                unsafe { intrinsics::simd_cast(self) }
            }
        }
    }
}

implement! { f32 }
implement! { f64 }
