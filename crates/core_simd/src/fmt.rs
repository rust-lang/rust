macro_rules! debug_wrapper {
    { $($trait:ident => $name:ident,)* } => {
        $(
            pub(crate) fn $name<T: core::fmt::$trait>(slice: &[T], f: &mut core::fmt::Formatter) -> core::fmt::Result {
                #[repr(transparent)]
                struct Wrapper<'a, T: core::fmt::$trait>(&'a T);

                impl<T: core::fmt::$trait> core::fmt::Debug for Wrapper<'_, T> {
                    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                        self.0.fmt(f)
                    }
                }

                f.debug_list()
                    .entries(slice.iter().map(|x| Wrapper(x)))
                    .finish()
            }
        )*
    }
}

debug_wrapper! {
    Debug => format,
    Binary => format_binary,
    LowerExp => format_lower_exp,
    UpperExp => format_upper_exp,
    Octal => format_octal,
    LowerHex => format_lower_hex,
    UpperHex => format_upper_hex,
}

macro_rules! impl_fmt_trait {
    { $($type:ty => $(($trait:ident, $format:ident)),*;)* } => {
        $( // repeat type
            $( // repeat trait
                impl core::fmt::$trait for $type {
                    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                        $format(self.as_ref(), f)
                    }
                }
            )*
        )*
    };
    { integers: $($type:ty,)* } => {
        impl_fmt_trait! {
            $($type =>
              (Debug, format),
              (Binary, format_binary),
              (LowerExp, format_lower_exp),
              (UpperExp, format_upper_exp),
              (Octal, format_octal),
              (LowerHex, format_lower_hex),
              (UpperHex, format_upper_hex);
            )*
        }
    };
    { floats: $($type:ty,)* } => {
        impl_fmt_trait! {
            $($type =>
              (Debug, format),
              (LowerExp, format_lower_exp),
              (UpperExp, format_upper_exp);
            )*
        }
    };
    { masks: $($type:ty,)* } => {
        impl_fmt_trait! {
            $($type =>
              (Debug, format);
            )*
        }
    }
}

impl_fmt_trait! {
    integers:
        crate::u8x8,    crate::u8x16,   crate::u8x32,   crate::u8x64,
        crate::i8x8,    crate::i8x16,   crate::i8x32,   crate::i8x64,
        crate::u16x4,   crate::u16x8,   crate::u16x16,  crate::u16x32,
        crate::i16x4,   crate::i16x8,   crate::i16x16,  crate::i16x32,
        crate::u32x2,   crate::u32x4,   crate::u32x8,   crate::u32x16,
        crate::i32x2,   crate::i32x4,   crate::i32x8,   crate::i32x16,
        crate::u64x2,   crate::u64x4,   crate::u64x8,
        crate::i64x2,   crate::i64x4,   crate::i64x8,
        crate::u128x2,  crate::u128x4,
        crate::i128x2,  crate::i128x4,
        crate::usizex2, crate::usizex4, crate::usizex8,
        crate::isizex2, crate::isizex4, crate::isizex8,
}

impl_fmt_trait! {
    floats:
        crate::f32x2, crate::f32x4, crate::f32x8, crate::f32x16,
        crate::f64x2, crate::f64x4, crate::f64x8,
}

impl_fmt_trait! {
    masks:
        crate::masks::wide::m8x8,    crate::masks::wide::m8x16,   crate::masks::wide::m8x32,   crate::masks::wide::m8x64,
        crate::masks::wide::m16x4,   crate::masks::wide::m16x8,   crate::masks::wide::m16x16,  crate::masks::wide::m16x32,
        crate::masks::wide::m32x2,   crate::masks::wide::m32x4,   crate::masks::wide::m32x8,   crate::masks::wide::m32x16,
        crate::masks::wide::m64x2,   crate::masks::wide::m64x4,   crate::masks::wide::m64x8,
        crate::masks::wide::m128x2,  crate::masks::wide::m128x4,
        crate::masks::wide::msizex2, crate::masks::wide::msizex4, crate::masks::wide::msizex8,
}
