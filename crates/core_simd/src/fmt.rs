macro_rules! impl_fmt_trait {
    { $($trait:ident,)* } => {
        $(
            impl<Element, const LANES: usize> core::fmt::$trait for crate::Simd<Element, LANES>
            where
                crate::LaneCount<LANES>: crate::SupportedLaneCount,
                Element: crate::SimdElement + core::fmt::$trait,
            {
                fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                    #[repr(transparent)]
                    struct Wrapper<'a, T: core::fmt::$trait>(&'a T);

                    impl<T: core::fmt::$trait> core::fmt::Debug for Wrapper<'_, T> {
                        fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                            self.0.fmt(f)
                        }
                    }

                    f.debug_list()
                        .entries(self.as_array().iter().map(|x| Wrapper(x)))
                        .finish()
                }
            }
        )*
    }
}

impl_fmt_trait! {
    Debug,
    Binary,
    LowerExp,
    UpperExp,
    Octal,
    LowerHex,
    UpperHex,
}
