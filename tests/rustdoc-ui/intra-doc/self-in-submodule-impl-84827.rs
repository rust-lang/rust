// https://github.com/rust-lang/rust/issues/84827
// Regression test: `Self` in intra-doc links inside `impl` blocks in submodules
// should resolve correctly even when the type is not directly in scope.

//@ check-pass
#![deny(rustdoc::broken_intra_doc_links)]

pub struct Foo {
    pub foo: i32,
}

impl Foo {
    pub fn method(&self) {}
}

pub mod bar {
    impl crate::Foo {
        /// Link to [`Self`], [`Self::foo`], [`Self::method`], and [`Self::baz`].
        pub fn baz(&self) {}
    }
}
