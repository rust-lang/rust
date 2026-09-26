//@ check-pass

pub mod inner {
    // [123](a::b::c)
    //#[deprecated = "[broken link](TypeAlias::x)"]
    #[doc(no_inline)]
    #[deprecated = "bar"]
    pub use std::vec::Vec;
    //#[deprecated = "[working link](std::range)"]
    #[deprecated = "foo"]
    pub use std::ops::Range;
}
