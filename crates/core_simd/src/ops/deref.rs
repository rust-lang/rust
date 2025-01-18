//! This module hacks in "implicit deref" for Simd's operators.
//! Ideally, Rust would take care of this itself,
//! and method calls usually handle the LHS implicitly.
//! But this is not the case with arithmetic ops.

use super::*;

macro_rules! deref_lhs {
    (impl<T, const N: usize> $trait:ident for $simd:ty {
            fn $call:ident
        }) => {
        impl<T, const N: usize> $trait<$simd> for &$simd
        where
            T: SimdElement,
            $simd: $trait<$simd, Output = $simd>,
            LaneCount<N>: SupportedLaneCount,
        {
            type Output = Simd<T, N>;

            #[inline]
            #[must_use = "operator returns a new vector without mutating the inputs"]
            fn $call(self, rhs: $simd) -> Self::Output {
                (*self).$call(rhs)
            }
        }
    };
}

macro_rules! deref_rhs {
    (impl<T, const N: usize> $trait:ident for $simd:ty {
            fn $call:ident
        }) => {
        impl<T, const N: usize> $trait<&$simd> for $simd
        where
            T: SimdElement,
            $simd: $trait<$simd, Output = $simd>,
            LaneCount<N>: SupportedLaneCount,
        {
            type Output = Simd<T, N>;

            #[inline]
            #[must_use = "operator returns a new vector without mutating the inputs"]
            fn $call(self, rhs: &$simd) -> Self::Output {
                self.$call(*rhs)
            }
        }
    };
}

macro_rules! deref_ops {
    ($(impl<T, const N: usize> $trait:ident for $simd:ty {
            fn $call:ident
        })*) => {
        $(
            deref_rhs! {
                impl<T, const N: usize> $trait for $simd {
                    fn $call
                }
            }
            deref_lhs! {
                impl<T, const N: usize> $trait for $simd {
                    fn $call
                }
            }
            impl<'lhs, 'rhs, T, const N: usize> $trait<&'rhs $simd> for &'lhs $simd
            where
                T: SimdElement,
                $simd: $trait<$simd, Output = $simd>,
                LaneCount<N>: SupportedLaneCount,
            {
                type Output = $simd;

                #[inline]
                #[must_use = "operator returns a new vector without mutating the inputs"]
                fn $call(self, rhs: &'rhs $simd) -> Self::Output {
                    (*self).$call(*rhs)
                }
            }
        )*
    }
}

deref_ops! {
    // Arithmetic
    impl<T, const N: usize> Add for Simd<T, N> {
        fn add
    }

    impl<T, const N: usize> Mul for Simd<T, N> {
        fn mul
    }

    impl<T, const N: usize> Sub for Simd<T, N> {
        fn sub
    }

    impl<T, const N: usize> Div for Simd<T, N> {
        fn div
    }

    impl<T, const N: usize> Rem for Simd<T, N> {
        fn rem
    }

    // Bitops
    impl<T, const N: usize> BitAnd for Simd<T, N> {
        fn bitand
    }

    impl<T, const N: usize> BitOr for Simd<T, N> {
        fn bitor
    }

    impl<T, const N: usize> BitXor for Simd<T, N> {
        fn bitxor
    }

    impl<T, const N: usize> Shl for Simd<T, N> {
        fn shl
    }

    impl<T, const N: usize> Shr for Simd<T, N> {
        fn shr
    }
}
