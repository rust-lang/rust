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
    { $($type:ident => $(($trait:ident, $format:ident)),*;)* } => {
        $( // repeat type
            $( // repeat trait
                impl<const LANES: usize> core::fmt::$trait for crate::$type<LANES>
                where
                    Self: crate::LanesAtMost32,
                {
                    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                        $format(self.as_ref(), f)
                    }
                }
            )*
        )*
    };
    { integers: $($type:ident,)* } => {
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
    { floats: $($type:ident,)* } => {
        impl_fmt_trait! {
            $($type =>
              (Debug, format),
              (LowerExp, format_lower_exp),
              (UpperExp, format_upper_exp);
            )*
        }
    };
    { masks: $($type:ident,)* } => {
        impl_fmt_trait! {
            $($type =>
              (Debug, format);
            )*
        }
    }
}

impl_fmt_trait! {
    integers:
        SimdU8, SimdU16, SimdU32, SimdU64, SimdU128,
        SimdI8, SimdI16, SimdI32, SimdI64, SimdI128,
        SimdUsize, SimdIsize,
}

impl_fmt_trait! {
    floats:
        SimdF32, SimdF64,
}
