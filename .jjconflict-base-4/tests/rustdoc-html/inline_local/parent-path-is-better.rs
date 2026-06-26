//! Test case for [134702]
//!
//! [134702]: https://github.com/rust-lang/rust/issues/134702
#![crate_name = "foo"]

pub mod inside1 {
    pub use self::inner::Inside1;
    mod inner {
        pub struct Inside1;
        impl Inside1 {
            pub fn stuff(self) {}
        }
    }
}

pub mod inside2 {
    pub use self::inner::Inside2;
    mod inner {
        pub struct Inside2;
        impl Inside2 {
            pub fn stuff(self) {}
        }
    }
}

pub mod nested {
    //! [Inside1] [Inside2]
    //@ has foo/nested/index.html '//a[@href="../struct.Inside1.html"]' 'Inside1'
    //@ has foo/nested/index.html '//a[@href="../struct.Inside2.html"]' 'Inside2'
    //! [Inside1::stuff] [Inside2::stuff]
    //@ has foo/nested/index.html '//a[@href="../struct.Inside1.html#method.stuff"]' 'Inside1::stuff'
    //@ has foo/nested/index.html '//a[@href="../struct.Inside2.html#method.stuff"]' 'Inside2::stuff'
    use crate::inside1::Inside1;
    use crate::inside2::Inside2;
}

#[doc(inline)]
pub use inside1::Inside1;
#[doc(inline)]
pub use inside2::Inside2;
