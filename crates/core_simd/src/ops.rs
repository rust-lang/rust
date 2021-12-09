use crate::simd::intrinsics;
use crate::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
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

/// SAFETY: This macro should not be used for anything except Shl or Shr, and passed the appropriate shift intrinsic.
/// It handles performing a bitand in addition to calling the shift operator, so that the result
/// is well-defined: LLVM can return a poison value if you shl, lshr, or ashr if rhs >= <Int>::BITS
/// At worst, this will maybe add another instruction and cycle,
/// at best, it may open up more optimization opportunities,
/// or simply be elided entirely, especially for SIMD ISAs which default to this.
///
// FIXME: Consider implementing this in cg_llvm instead?
// cg_clif defaults to this, and scalar MIR shifts also default to wrapping
macro_rules! wrap_bitshift_inner {
    (impl<const LANES: usize> $op:ident for Simd<$int:ty, LANES> {
        fn $call:ident(self, rhs: Self) -> Self::Output {
            unsafe { $simd_call:ident }
        }
    }) => {
        impl<const LANES: usize> $op for Simd<$int, LANES>
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
        }
    };
}

macro_rules! wrap_bitshifts {
    ($(impl<const LANES: usize> ShiftOps for Simd<$int:ty, LANES> {
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
     })*) => {
        $(
            wrap_bitshift_inner! {
                impl<const LANES: usize> Shl for Simd<$int, LANES> {
                    fn shl(self, rhs: Self) -> Self::Output {
                        unsafe { simd_shl }
                    }
                }
            }
            wrap_bitshift_inner! {
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

wrap_bitshifts! {
    impl<const LANES: usize> ShiftOps for Simd<i8, LANES> {
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> ShiftOps for Simd<i16, LANES> {
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> ShiftOps for Simd<i32, LANES> {
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> ShiftOps for Simd<i64, LANES> {
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> ShiftOps for Simd<isize, LANES> {
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> ShiftOps for Simd<u8, LANES> {
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> ShiftOps for Simd<u16, LANES> {
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> ShiftOps for Simd<u32, LANES> {
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> ShiftOps for Simd<u64, LANES> {
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }

    impl<const LANES: usize> ShiftOps for Simd<usize, LANES> {
        fn shl(self, rhs: Self) -> Self::Output;
        fn shr(self, rhs: Self) -> Self::Output;
    }
}

/// Automatically implements operators over references in addition to the provided operator.
macro_rules! impl_ref_ops {
    // binary op
    {
        impl<const $lanes:ident: usize> core::ops::$trait:ident<$rhs:ty> for $type:ty
        where
            LaneCount<$lanes2:ident>: SupportedLaneCount,
        {
            type Output = $output:ty;

            $(#[$attrs:meta])*
            fn $fn:ident($self_tok:ident, $rhs_arg:ident: $rhs_arg_ty:ty) -> Self::Output $body:tt
        }
    } => {
        impl<const $lanes: usize> core::ops::$trait<$rhs> for $type
        where
            LaneCount<$lanes2>: SupportedLaneCount,
        {
            type Output = $output;

            $(#[$attrs])*
            fn $fn($self_tok, $rhs_arg: $rhs_arg_ty) -> Self::Output $body
        }
    };
}

/// Automatically implements operators over vectors and scalars for a particular vector.
macro_rules! impl_op {
    { impl Add for $scalar:ty } => {
        impl_op! { @binary $scalar, Add::add, simd_add }
    };
    { impl Sub for $scalar:ty } => {
        impl_op! { @binary $scalar, Sub::sub, simd_sub }
    };
    { impl Mul for $scalar:ty } => {
        impl_op! { @binary $scalar, Mul::mul, simd_mul }
    };
    { impl Div for $scalar:ty } => {
        impl_op! { @binary $scalar, Div::div, simd_div }
    };
    { impl Rem for $scalar:ty } => {
        impl_op! { @binary $scalar, Rem::rem, simd_rem }
    };
    { impl BitAnd for $scalar:ty } => {
        impl_op! { @binary $scalar, BitAnd::bitand, simd_and }
    };
    { impl BitOr for $scalar:ty } => {
        impl_op! { @binary $scalar, BitOr::bitor, simd_or }
    };
    { impl BitXor for $scalar:ty } => {
        impl_op! { @binary $scalar, BitXor::bitxor, simd_xor }
    };

    // generic binary op with assignment when output is `Self`
    { @binary $scalar:ty, $trait:ident :: $trait_fn:ident, $intrinsic:ident } => {
        impl_ref_ops! {
            impl<const LANES: usize> core::ops::$trait<Self> for Simd<$scalar, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                type Output = Self;

                #[inline]
                fn $trait_fn(self, rhs: Self) -> Self::Output {
                    unsafe {
                        intrinsics::$intrinsic(self, rhs)
                    }
                }
            }
        }
    };
}

/// Implements floating-point operators for the provided types.
macro_rules! impl_float_ops {
    { $($scalar:ty),* } => {
        $(
            impl_op! { impl Add for $scalar }
            impl_op! { impl Sub for $scalar }
            impl_op! { impl Mul for $scalar }
            impl_op! { impl Div for $scalar }
            impl_op! { impl Rem for $scalar }
        )*
    };
}

/// Implements unsigned integer operators for the provided types.
macro_rules! impl_unsigned_int_ops {
    { $($scalar:ty),* } => {
        $(
            impl_op! { impl Add for $scalar }
            impl_op! { impl Sub for $scalar }
            impl_op! { impl Mul for $scalar }
            impl_op! { impl BitAnd for $scalar }
            impl_op! { impl BitOr  for $scalar }
            impl_op! { impl BitXor for $scalar }

            // Integers panic on divide by 0
            impl_ref_ops! {
                impl<const LANES: usize> core::ops::Div<Self> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    type Output = Self;

                    #[inline]
                    fn div(self, rhs: Self) -> Self::Output {
                        if rhs.as_array()
                            .iter()
                            .any(|x| *x == 0)
                        {
                            panic!("attempt to divide by zero");
                        }

                        // Guards for div(MIN, -1),
                        // this check only applies to signed ints
                        if <$scalar>::MIN != 0 && self.as_array().iter()
                                .zip(rhs.as_array().iter())
                                .any(|(x,y)| *x == <$scalar>::MIN && *y == -1 as _) {
                            panic!("attempt to divide with overflow");
                        }
                        unsafe { intrinsics::simd_div(self, rhs) }
                    }
                }
            }

            // remainder panics on zero divisor
            impl_ref_ops! {
                impl<const LANES: usize> core::ops::Rem<Self> for Simd<$scalar, LANES>
                where
                    LaneCount<LANES>: SupportedLaneCount,
                {
                    type Output = Self;

                    #[inline]
                    fn rem(self, rhs: Self) -> Self::Output {
                        if rhs.as_array()
                            .iter()
                            .any(|x| *x == 0)
                        {
                            panic!("attempt to calculate the remainder with a divisor of zero");
                        }

                        // Guards for rem(MIN, -1)
                        // this branch applies the check only to signed ints
                        if <$scalar>::MIN != 0 && self.as_array().iter()
                                .zip(rhs.as_array().iter())
                                .any(|(x,y)| *x == <$scalar>::MIN && *y == -1 as _) {
                            panic!("attempt to calculate the remainder with overflow");
                        }
                        unsafe { intrinsics::simd_rem(self, rhs) }
                    }
                }
            }
        )*
    };
}

/// Implements unsigned integer operators for the provided types.
macro_rules! impl_signed_int_ops {
    { $($scalar:ty),* } => {
        impl_unsigned_int_ops! { $($scalar),* }
    };
}

impl_unsigned_int_ops! { u8, u16, u32, u64, usize }
impl_signed_int_ops! { i8, i16, i32, i64, isize }
impl_float_ops! { f32, f64 }
