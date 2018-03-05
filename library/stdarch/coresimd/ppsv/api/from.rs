//! Implements the From trait for vector types, which performs a lane-wise
//! cast vector types with the same number of lanes.

macro_rules! impl_from {
    ($to:ident: $elem_ty:ident, $test_mod:ident | $($from:ident),+) => {
        $(
            impl From<::simd::$from> for $to {
                #[inline]
                fn from(f: ::simd::$from) -> $to {
                    unsafe { simd_cast(f) }
                }
            }
        )+

        #[cfg(test)]
        mod $test_mod {
            $(
                #[test]
                fn $from() {
                    use ::std::convert::{From, Into};
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
    }
}
