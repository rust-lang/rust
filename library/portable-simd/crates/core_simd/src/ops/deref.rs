//! This module hacks in "implicit deref" for Simd's operators.
//! Ideally, Rust would take care of this itself,
//! and method calls usually handle the LHS implicitly.
//! But this is not the case with arithmetic ops.
use super::*;

macro_rules! deref_lhs {
    (impl<T, const LANES: usize> $trait:ident for $simd:ty {
            fn $call:ident
        }) => {
        impl<T, const LANES: usize> $trait<$simd> for &$simd
        where
            T: SimdElement,
            $simd: $trait<$simd, Output = $simd>,
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Output = Simd<T, LANES>;

            #[inline]
            #[must_use = "operator returns a new vector without mutating the inputs"]
            fn $call(self, rhs: $simd) -> Self::Output {
                (*self).$call(rhs)
            }
        }
    };
}

macro_rules! deref_rhs {
    (impl<T, const LANES: usize> $trait:ident for $simd:ty {
            fn $call:ident
        }) => {
        impl<T, const LANES: usize> $trait<&$simd> for $simd
        where
            T: SimdElement,
            $simd: $trait<$simd, Output = $simd>,
            LaneCount<LANES>: SupportedLaneCount,
        {
            type Output = Simd<T, LANES>;

            #[inline]
            #[must_use = "operator returns a new vector without mutating the inputs"]
            fn $call(self, rhs: &$simd) -> Self::Output {
                self.$call(*rhs)
            }
        }
    };
}

macro_rules! deref_ops {
    ($(impl<T, const LANES: usize> $trait:ident for $simd:ty {
            fn $call:ident
        })*) => {
        $(
            deref_rhs! {
                impl<T, const LANES: usize> $trait for $simd {
                    fn $call
                }
            }
            deref_lhs! {
                impl<T, const LANES: usize> $trait for $simd {
                    fn $call
                }
            }
            impl<'lhs, 'rhs, T, const LANES: usize> $trait<&'rhs $simd> for &'lhs $simd
            where
                T: SimdElement,
                $simd: $trait<$simd, Output = $simd>,
                LaneCount<LANES>: SupportedLaneCount,
            {
                type Output = $simd;

                #[inline]
                #[must_use = "operator returns a new vector without mutating the inputs"]
                fn $call(self, rhs: &$simd) -> Self::Output {
                    (*self).$call(*rhs)
                }
            }
        )*
    }
}

deref_ops! {
    // Arithmetic
    impl<T, const LANES: usize> Add for Simd<T, LANES> {
        fn add
    }

    impl<T, const LANES: usize> Mul for Simd<T, LANES> {
        fn mul
    }

    impl<T, const LANES: usize> Sub for Simd<T, LANES> {
        fn sub
    }

    impl<T, const LANES: usize> Div for Simd<T, LANES> {
        fn div
    }

    impl<T, const LANES: usize> Rem for Simd<T, LANES> {
        fn rem
    }

    // Bitops
    impl<T, const LANES: usize> BitAnd for Simd<T, LANES> {
        fn bitand
    }

    impl<T, const LANES: usize> BitOr for Simd<T, LANES> {
        fn bitor
    }

    impl<T, const LANES: usize> BitXor for Simd<T, LANES> {
        fn bitxor
    }

    impl<T, const LANES: usize> Shl for Simd<T, LANES> {
        fn shl
    }

    impl<T, const LANES: usize> Shr for Simd<T, LANES> {
        fn shr
    }
}
