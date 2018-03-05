//! Implements `PartialEq` for vector types.

macro_rules! impl_partial_eq {
    ($id:ident) => {
        impl PartialEq<$id> for $id {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                $id::eq(*self, *other).all()
            }
            #[inline]
            fn ne(&self, other: &Self) -> bool {
                $id::ne(*self, *other).all()
            }
        }
    }
}

#[cfg(test)]
#[macro_export]
macro_rules! test_partial_eq {
    ($id:ident, $true:expr, $false:expr) => {
        #[test]
        fn partial_eq() {
            use ::coresimd::simd::*;

            let a = $id::splat($false);
            let b = $id::splat($true);

            assert!(a != b);
            assert!(!(a == b));
            assert!(a == a);
            assert!(!(a != a));
        }
    }
}
