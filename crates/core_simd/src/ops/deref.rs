//! This module hacks in "implicit deref" for Simd's operators.
//! Ideally, Rust would take care of this itself,
//! and method calls usually handle the LHS implicitly.
//! So, we'll manually deref the RHS.
use super::*;

macro_rules! deref_ops {
    ($(impl<T, const LANES: usize> $trait:ident<&Self> for Simd<T, LANES> {
            fn $call:ident(rhs: &Self)
        })*) => {
        $(impl<T, const LANES: usize> $trait<&Self> for Simd<T, LANES>
        where
            Self: $trait<Self, Output = Self>,
            T: SimdElement,
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Output = Self;

            #[inline]
            #[must_use = "operator returns a new vector without mutating the inputs"]
            fn $call(self, rhs: &Self) -> Self::Output {
                self.$call(*rhs)
            }
        })*
    }
}

deref_ops! {
    // Arithmetic
    impl<T, const LANES: usize> Add<&Self> for Simd<T, LANES> {
        fn add(rhs: &Self)
    }

    impl<T, const LANES: usize> Mul<&Self> for Simd<T, LANES> {
        fn mul(rhs: &Self)
    }

    impl<T, const LANES: usize> Sub<&Self> for Simd<T, LANES> {
        fn sub(rhs: &Self)
    }

    impl<T, const LANES: usize> Div<&Self> for Simd<T, LANES> {
        fn div(rhs: &Self)
    }

    impl<T, const LANES: usize> Rem<&Self> for Simd<T, LANES> {
        fn rem(rhs: &Self)
    }

    // Bitops
    impl<T, const LANES: usize> BitAnd<&Self> for Simd<T, LANES> {
        fn bitand(rhs: &Self)
    }

    impl<T, const LANES: usize> BitOr<&Self> for Simd<T, LANES> {
        fn bitor(rhs: &Self)
    }

    impl<T, const LANES: usize> BitXor<&Self> for Simd<T, LANES> {
        fn bitxor(rhs: &Self)
    }

    impl<T, const LANES: usize> Shl<&Self> for Simd<T, LANES> {
        fn shl(rhs: &Self)
    }

    impl<T, const LANES: usize> Shr<&Self> for Simd<T, LANES> {
        fn shr(rhs: &Self)
    }
}
