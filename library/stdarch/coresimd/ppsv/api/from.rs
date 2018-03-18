//! Implements the From trait for vector types, which performs a lane-wise
//! cast vector types with the same number of lanes.
#![allow(unused)]

macro_rules! impl_from_impl {
    ($from:ident, $to:ident) => {
        impl ::convert::From<::simd::$from> for $to {
            #[inline]
            fn from(f: ::simd::$from) -> $to {
                use coresimd::simd_llvm::simd_cast;
                unsafe { simd_cast(f) }
            }
        }
    }
}

macro_rules! impl_from_ {
    ($to:ident, $from:ident) => {
        vector_impl!([impl_from_impl, $to, $from]);
    }
}

macro_rules! impl_from {
    ($to:ident: $elem_ty:ident, $test_mod:ident, $test_macro:ident | $($from:ident),+) => {
        $(
            impl_from_!($from, $to);
        )+

        $test_macro!(
            #[cfg(test)]
            mod $test_mod {
                $(
                    #[test]
                    fn $from() {
                        use std::convert::{From, Into};
                        use ::coresimd::simd::{$from, $to};
                        use ::std::default::Default;
                        assert_eq!($to::lanes(), $from::lanes());
                        let a: $from = $from::default();
                        let b_0: $to = From::from(a);
                        let b_1: $to = a.into();
                        assert_eq!(b_0, b_1);
                    }
                )+
            }
        );
    }
}
