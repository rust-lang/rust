// Restricted glob must not stop `f` from being encoded.

mod inner {
    pub fn f() -> u32 {
        42
    }
}

mod facade {
    #[allow(unused_imports)]
    pub(crate) use super::inner::f;
}

#[allow(unused_imports)]
use facade::*;
pub use inner::*;
