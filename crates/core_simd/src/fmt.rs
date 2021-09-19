use crate::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
use core::fmt;

macro_rules! impl_fmt_trait {
    { $($trait:ident,)* } => {
        $(
            impl<T, const LANES: usize> fmt::$trait for Simd<T, LANES>
            where
                LaneCount<LANES>: SupportedLaneCount,
                T: SimdElement + fmt::$trait,
            {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    #[repr(transparent)]
                    struct Wrapper<'a, T: fmt::$trait>(&'a T);

                    impl<T: fmt::$trait> fmt::Debug for Wrapper<'_, T> {
                        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
