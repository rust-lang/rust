//! Implements `PartialEq` for vector types.
#![allow(unused)]

macro_rules! impl_partial_eq {
    ($id:ident) => {
        impl ::cmp::PartialEq<$id> for $id {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                $id::eq(*self, *other).all()
            }
            #[inline]
            fn ne(&self, other: &Self) -> bool {
                $id::ne(*self, *other).any()
            }
        }
    };
}

#[cfg(test)]
macro_rules! test_partial_eq {
    ($id:ident, $true:expr, $false:expr) => {
        #[test]
        fn partial_eq() {
            use coresimd::simd::*;

            let a = $id::splat($false);
            let b = $id::splat($true);

            assert!(a != b);
            assert!(!(a == b));
            assert!(a == a);
            assert!(!(a != a));

            // Test further to make sure comparisons work with non-splatted
            // values.
            // This is to test the fix for #511

            let a = $id::splat($false).replace(0, $true);
            let b = $id::splat($true);

            assert!(a != b);
            assert!(!(a == b));
            assert!(a == a);
            assert!(!(a != a));
        }
    };
}
