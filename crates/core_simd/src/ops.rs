use crate::simd::{LaneCount, Mask, Simd, SimdElement, SupportedLaneCount};
use core::ops::{Add, Mul};
use core::ops::{BitAnd, BitOr, BitXor};
use core::ops::{Div, Rem, Sub};
use core::ops::{Shl, Shr};

mod assign;
mod deref;
mod unary;

impl<I, T, const LANES: usize> core::ops::Index<I> for Simd<T, LANES>
where
    T: SimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    I: core::slice::SliceIndex<[T]>,
{
    type Output = I::Output;
    fn index(&self, index: I) -> &Self::Output {
        &self.as_array()[index]
    }
}

impl<I, T, const LANES: usize> core::ops::IndexMut<I> for Simd<T, LANES>
where
    T: SimdElement,
    LaneCount<LANES>: SupportedLaneCount,
    I: core::slice::SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.as_mut_array()[index]
    }
}

macro_rules! unsafe_base_op {
    ($(impl<const LANES: usize> $op:ident for Simd<$scalar:ty, LANES> {
        fn $call:ident(self, rhs: Self) -> Self::Output {
            unsafe{ $simd_call:ident }
        }
    })*) => {
        $(impl<const LANES: usize> $op for Simd<$scalar, LANES>
            where
                $scalar: SimdElement,
                LaneCount<LANES>: SupportedLaneCount,
            {
                type Output = Self;

                #[inline]
                #[must_use = "operator returns a new vector without mutating the inputs"]
                fn $call(self, rhs: Self) -> Self::Output {
                    unsafe { $crate::intrinsics::$simd_call(self, rhs) }
                }
            }
        )*
    }
}

/// SAFETY: This macro should not be used for anything except Shl or Shr, and passed the appropriate shift intrinsic.
/// It handles performing a bitand in addition to calling the shift operator, so that the result
/// is well-defined: LLVM can return a poison value if you shl, lshr, or ashr if rhs >= <Int>::BITS
/// At worst, this will maybe add another instruction and cycle,
/// at best, it may open up more optimization opportunities,
/// or simply be elided entirely, especially for SIMD ISAs which default to this.
///
// FIXME: Consider implementing this in cg_llvm instead?
// cg_clif defaults to this, and scalar MIR shifts also default to wrapping
macro_rules! wrap_bitshift {
    ($(impl<const LANES: usize> $op:ident for Simd<$int:ty, LANES> {
        fn $call:ident(self, rhs: Self) -> Self::Output {
            unsafe { $simd_call:ident }
        }
    })*) => {
        $(impl<const LANES: usize> $op for Simd<$int, LANES>
        where
            $int: SimdElement,
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Output = Self;

            #[inline]
            #[must_use = "operator returns a new vector without mutating the inputs"]
            fn $call(self, rhs: Self) -> Self::Output {
                unsafe {
                    $crate::intrinsics::$simd_call(self, rhs.bitand(Simd::splat(<$int>::BITS as $int - 1)))
                }
            }
        })*
    };
}

macro_rules! bitops {
    ($(impl<const LANES: usize> BitOps for Simd<$int:ty, LANES> {
        fn bitand(self, rhs: Self) -> Self::Output;
        fn bitor(self, rhs: Self) -> Self::Output;
        fn bitxor(self, rhs: Self) -> Self::Output;
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
     })*) => {
        $(
            unsafe_base_op!{
                impl<const LANES: usize> BitAnd for Simd<$int, LANES> {
                    fn bitand(self, rhs: Self) -> Self::Output {
                        unsafe { simd_and }
                    }
                }

                impl<const LANES: usize> BitOr for Simd<$int, LANES> {
                    fn bitor(self, rhs: Self) -> Self::Output {
                        unsafe { simd_or }
                    }
                }

                impl<const LANES: usize> BitXor for Simd<$int, LANES> {
                    fn bitxor(self, rhs: Self) -> Self::Output {
                        unsafe { simd_xor }
                    }
                }
            }
            wrap_bitshift! {
                impl<const LANES: usize> Shl for Simd<$int, LANES> {
                    fn shl(self, rhs: Self) -> Self::Output {
                        unsafe { simd_shl }
                    }
                }

                impl<const LANES: usize> Shr for Simd<$int, LANES> {
                    fn shr(self, rhs: Self) -> Self::Output {
                        // This automatically monomorphizes to lshr or ashr, depending,
                        // so it's fine to use it for both UInts and SInts.
                        unsafe { simd_shr }
                    }
                }
            }
        )*
    };
}

// Integers can always accept bitand, bitor, and bitxor.
// The only question is how to handle shifts >= <Int>::BITS?
// Our current solution uses wrapping logic.
bitops! {
    impl<const LANES: usize> BitOps for Simd<i8, LANES> {
        fn bitand(self, rhs: Self) -> Self::Output;
        fn bitor(self, rhs: Self) -> Self::Output;
        fn bitxor(self, rhs: Self) -> Self::Output;
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> BitOps for Simd<i16, LANES> {
        fn bitand(self, rhs: Self) -> Self::Output;
        fn bitor(self, rhs: Self) -> Self::Output;
        fn bitxor(self, rhs: Self) -> Self::Output;
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> BitOps for Simd<i32, LANES> {
        fn bitand(self, rhs: Self) -> Self::Output;
        fn bitor(self, rhs: Self) -> Self::Output;
        fn bitxor(self, rhs: Self) -> Self::Output;
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> BitOps for Simd<i64, LANES> {
        fn bitand(self, rhs: Self) -> Self::Output;
        fn bitor(self, rhs: Self) -> Self::Output;
        fn bitxor(self, rhs: Self) -> Self::Output;
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> BitOps for Simd<isize, LANES> {
        fn bitand(self, rhs: Self) -> Self::Output;
        fn bitor(self, rhs: Self) -> Self::Output;
        fn bitxor(self, rhs: Self) -> Self::Output;
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> BitOps for Simd<u8, LANES> {
        fn bitand(self, rhs: Self) -> Self::Output;
        fn bitor(self, rhs: Self) -> Self::Output;
        fn bitxor(self, rhs: Self) -> Self::Output;
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> BitOps for Simd<u16, LANES> {
        fn bitand(self, rhs: Self) -> Self::Output;
        fn bitor(self, rhs: Self) -> Self::Output;
        fn bitxor(self, rhs: Self) -> Self::Output;
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> BitOps for Simd<u32, LANES> {
        fn bitand(self, rhs: Self) -> Self::Output;
        fn bitor(self, rhs: Self) -> Self::Output;
        fn bitxor(self, rhs: Self) -> Self::Output;
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> BitOps for Simd<u64, LANES> {
        fn bitand(self, rhs: Self) -> Self::Output;
        fn bitor(self, rhs: Self) -> Self::Output;
        fn bitxor(self, rhs: Self) -> Self::Output;
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> BitOps for Simd<usize, LANES> {
        fn bitand(self, rhs: Self) -> Self::Output;
        fn bitor(self, rhs: Self) -> Self::Output;
        fn bitxor(self, rhs: Self) -> Self::Output;
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }
}

macro_rules! float_arith {
    ($(impl<const LANES: usize> FloatArith for Simd<$float:ty, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
     })*) => {
        $(
            unsafe_base_op!{
                impl<const LANES: usize> Add for Simd<$float, LANES> {
                    fn add(self, rhs: Self) -> Self::Output {
                        unsafe { simd_add }
                    }
                }

                impl<const LANES: usize> Mul for Simd<$float, LANES> {
                    fn mul(self, rhs: Self) -> Self::Output {
                        unsafe { simd_mul }
                    }
                }

                impl<const LANES: usize> Sub for Simd<$float, LANES> {
                    fn sub(self, rhs: Self) -> Self::Output {
                        unsafe { simd_sub }
                    }
                }

                impl<const LANES: usize> Div for Simd<$float, LANES> {
                    fn div(self, rhs: Self) -> Self::Output {
                        unsafe { simd_div }
                    }
                }

                impl<const LANES: usize> Rem for Simd<$float, LANES> {
                    fn rem(self, rhs: Self) -> Self::Output {
                        unsafe { simd_rem }
                    }
                }
            }
        )*
    };
}

// We don't need any special precautions here:
// Floats always accept arithmetic ops, but may become NaN.
float_arith! {
    impl<const LANES: usize> FloatArith for Simd<f32, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> FloatArith for Simd<f64, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
    }
}

// Division by zero is poison, according to LLVM.
// So is dividing the MIN value of a signed integer by -1,
// since that would return MAX + 1.
// FIXME: Rust allows <SInt>::MIN / -1,
// so we should probably figure out how to make that safe.
macro_rules! int_divrem_guard {
    ($(impl<const LANES: usize> $op:ident for Simd<$sint:ty, LANES> {
        const PANIC_ZERO: &'static str = $zero:literal;
        const PANIC_OVERFLOW: &'static str = $overflow:literal;
        fn $call:ident {
            unsafe { $simd_call:ident }
        }
    })*) => {
        $(impl<const LANES: usize> $op for Simd<$sint, LANES>
        where
            $sint: SimdElement,
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Output = Self;
            #[inline]
            #[must_use = "operator returns a new vector without mutating the inputs"]
            fn $call(self, rhs: Self) -> Self::Output {
                if rhs.lanes_eq(Simd::splat(0)).any() {
                    panic!("attempt to calculate the remainder with a divisor of zero");
                } else if <$sint>::MIN != 0 && self.lanes_eq(Simd::splat(<$sint>::MIN)) & rhs.lanes_eq(Simd::splat(-1 as _))
                    != Mask::splat(false)
                 {
                    panic!("attempt to calculate the remainder with overflow");
                } else {
                    unsafe { $crate::intrinsics::$simd_call(self, rhs) }
                 }
             }
        })*
    };
}

macro_rules! int_arith {
    ($(impl<const LANES: usize> IntArith for Simd<$sint:ty, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
    })*) => {
        $(
        unsafe_base_op!{
            impl<const LANES: usize> Add for Simd<$sint, LANES> {
                fn add(self, rhs: Self) -> Self::Output {
                    unsafe { simd_add }
                }
            }

            impl<const LANES: usize> Mul for Simd<$sint, LANES> {
                fn mul(self, rhs: Self) -> Self::Output {
                    unsafe { simd_mul }
                }
            }

            impl<const LANES: usize> Sub for Simd<$sint, LANES> {
                fn sub(self, rhs: Self) -> Self::Output {
                    unsafe { simd_sub }
                }
            }
        }

        int_divrem_guard!{
            impl<const LANES: usize> Div for Simd<$sint, LANES> {
                const PANIC_ZERO: &'static str = "attempt to divide by zero";
                const PANIC_OVERFLOW: &'static str = "attempt to divide with overflow";
                fn div {
                    unsafe { simd_div }
                }
            }

            impl<const LANES: usize> Rem for Simd<$sint, LANES> {
                const PANIC_ZERO: &'static str = "attempt to calculate the remainder with a divisor of zero";
                const PANIC_OVERFLOW: &'static str = "attempt to calculate the remainder with overflow";
                fn rem {
                    unsafe { simd_rem }
                }
            }
        })*
    }
}

int_arith! {
    impl<const LANES: usize> IntArith for Simd<i8, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> IntArith for Simd<i16, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> IntArith for Simd<i32, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> IntArith for Simd<i64, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> IntArith for Simd<isize, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> IntArith for Simd<u8, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> IntArith for Simd<u16, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> IntArith for Simd<u32, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> IntArith for Simd<u64, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> IntArith for Simd<usize, LANES> {
        fn add(self, rhs: Self) -> Self::Output;
        fn mul(self, rhs: Self) -> Self::Output;
        fn sub(self, rhs: Self) -> Self::Output;
        fn div(self, rhs: Self) -> Self::Output;
        fn rem(self, rhs: Self) -> Self::Output;
    }
}
