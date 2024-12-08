//! Assignment operators

use super::*;
use core::ops::{AddAssign, MulAssign}; // commutative binary op-assignment
use core::ops::{BitAndAssign, BitOrAssign, BitXorAssign}; // commutative bit binary op-assignment
use core::ops::{DivAssign, RemAssign, SubAssign}; // non-commutative binary op-assignment
use core::ops::{ShlAssign, ShrAssign}; // non-commutative bit binary op-assignment

// Arithmetic

macro_rules! assign_ops {
    ($(impl<T, U, const N: usize> $assignTrait:ident<U> for Simd<T, N>
        where
            Self: $trait:ident,
        {
            fn $assign_call:ident(rhs: U) {
                $call:ident
            }
        })*) => {
        $(impl<T, U, const N: usize> $assignTrait<U> for Simd<T, N>
        where
            Self: $trait<U, Output = Self>,
            T: SimdElement,
            LaneCount<N>: SupportedLaneCount,
        {
            #[inline]
            fn $assign_call(&mut self, rhs: U) {
                *self = self.$call(rhs);
            }
        })*
    }
}

assign_ops! {
    // Arithmetic
    impl<T, U, const N: usize> AddAssign<U> for Simd<T, N>
    where
        Self: Add,
    {
        fn add_assign(rhs: U) {
            add
        }
    }

    impl<T, U, const N: usize> MulAssign<U> for Simd<T, N>
    where
        Self: Mul,
    {
        fn mul_assign(rhs: U) {
            mul
        }
    }

    impl<T, U, const N: usize> SubAssign<U> for Simd<T, N>
    where
        Self: Sub,
    {
        fn sub_assign(rhs: U) {
            sub
        }
    }

    impl<T, U, const N: usize> DivAssign<U> for Simd<T, N>
    where
        Self: Div,
    {
        fn div_assign(rhs: U) {
            div
        }
    }
    impl<T, U, const N: usize> RemAssign<U> for Simd<T, N>
    where
        Self: Rem,
    {
        fn rem_assign(rhs: U) {
            rem
        }
    }

    // Bitops
    impl<T, U, const N: usize> BitAndAssign<U> for Simd<T, N>
    where
        Self: BitAnd,
    {
        fn bitand_assign(rhs: U) {
            bitand
        }
    }

    impl<T, U, const N: usize> BitOrAssign<U> for Simd<T, N>
    where
        Self: BitOr,
    {
        fn bitor_assign(rhs: U) {
            bitor
        }
    }

    impl<T, U, const N: usize> BitXorAssign<U> for Simd<T, N>
    where
        Self: BitXor,
    {
        fn bitxor_assign(rhs: U) {
            bitxor
        }
    }

    impl<T, U, const N: usize> ShlAssign<U> for Simd<T, N>
    where
        Self: Shl,
    {
        fn shl_assign(rhs: U) {
            shl
        }
    }

    impl<T, U, const N: usize> ShrAssign<U> for Simd<T, N>
    where
        Self: Shr,
    {
        fn shr_assign(rhs: U) {
            shr
        }
    }
}
