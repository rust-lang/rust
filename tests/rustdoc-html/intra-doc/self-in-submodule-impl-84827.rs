// https://github.com/rust-lang/rust/issues/84827
// Regression test: `Self` in intra-doc links inside `impl` blocks in submodules
// should produce correct hrefs in generated HTML.

#![crate_name = "foo"]
#![deny(rustdoc::broken_intra_doc_links)]

pub struct Foo {
    pub foo: i32,
}

impl Foo {
    pub fn method(&self) {}
}

pub mod bar {
    impl crate::Foo {
        //@ has foo/struct.Foo.html '//a[@href="struct.Foo.html"]' 'Self'
        //@ has foo/struct.Foo.html '//a[@href="struct.Foo.html#structfield.foo"]' 'Self::foo'
        //@ has foo/struct.Foo.html '//a[@href="struct.Foo.html#method.method"]' 'Self::method'
        //@ has foo/struct.Foo.html '//a[@href="struct.Foo.html#method.baz"]' 'Self::baz'
        /// Link to [`Self`], [`Self::foo`], [`Self::method`], and [`Self::baz`].
        pub fn baz(&self) {}
    }
}
