//! Test case for [134702]
//!
//! [134702]: https://github.com/rust-lang/rust/issues/134702
#![crate_name = "foo"]
#![stable(since = "1.0", feature = "v1")]

#![feature(staged_api, rustc_attrs)]

#[stable(since = "1.0", feature = "stb1")]
pub mod stb1 {
    #[doc(inline)]
    #[stable(since = "1.0", feature = "stb1")]
    pub use crate::uns::Inside1;
}

#[unstable(feature = "uns", issue = "135003")]
pub mod uns {
    #[stable(since = "1.0", feature = "stb1")]
    #[rustc_allowed_through_unstable_modules = "use stable path instead"]
    pub struct Inside1;
    #[stable(since = "1.0", feature = "stb2")]
    #[rustc_allowed_through_unstable_modules = "use stable path instead"]
    pub struct Inside2;
}

#[stable(since = "1.0", feature = "stb2")]
pub mod stb2 {
    #[doc(inline)]
    #[stable(since = "1.0", feature = "stb2")]
    pub use crate::uns::Inside2;
}

#[stable(since = "1.0", feature = "nested")]
pub mod nested {
    //! [Inside1] [Inside2]
    //@ has foo/nested/index.html '//a[@href="../stb1/struct.Inside1.html"]' 'Inside1'
    //@ has foo/nested/index.html '//a[@href="../stb2/struct.Inside2.html"]' 'Inside2'
    use crate::stb1::Inside1;
    use crate::stb2::Inside2;
}
