use crate::fmt::{Debug, Formatter, Result};

macro_rules! floating {
    ($($ty:ident)*) => {
        $(
            #[stable(feature = "rust1", since = "1.0.0")]
            impl Debug for $ty {
                #[inline]
                fn fmt(&self, _fmt: &mut Formatter<'_>) -> Result {
                    panic!("floating point fmt support is turned off");
                }
            }
        )*
    };
}

floating! { f16 f32 f64 f128 }
