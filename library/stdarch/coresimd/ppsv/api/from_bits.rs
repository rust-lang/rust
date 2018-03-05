//! Implements the `FromBits` trait for vector types, which performs bitwise
//! lossless transmutes between equally-sized vector types.

macro_rules! impl_from_bits_ {
    ($to:ident: $($from:ident),+) => {
        $(
            impl ::simd::FromBits<$from> for $to {
                #[inline]
                fn from_bits(f: $from) -> $to {
                    unsafe { mem::transmute(f) }
                }
            }
        )+
    }
}

macro_rules! impl_from_bits {
    ($to:ident: $elem_ty:ident, $test_mod:ident | $($from:ident),+) => {
        impl_from_bits_!($to: $($from),+);

        #[cfg(test)]
        mod $test_mod {
            $(
                #[test]
                fn $from() {
                    use ::coresimd::simd::{$from, $to, FromBits, IntoBits};
                    use ::std::{mem, default};
                    use default::Default;
                    assert_eq!(mem::size_of::<$from>(),
                               mem::size_of::<$to>());
                    let a: $from = $from::default();
                    let b_0: $to = FromBits::from_bits(a);
                    let b_1: $to = a.into_bits();
                    assert_eq!(b_0, b_1);
                }
            )+
        }
    }
}
