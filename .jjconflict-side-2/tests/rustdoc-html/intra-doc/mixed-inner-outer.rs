// This test ensures that when a documentation is composed of both inner and outer
// doc comments, the intra-doc link resolution still works as expected.
// Regression test for <https://github.com/rust-lang/rust/issues/78591>.
// Regression test for <https://github.com/rust-lang/rust/issues/119965>.
// Regression test for <https://github.com/rust-lang/rust/issues/134904>.

#![crate_name = "foo"]
#![deny(rustdoc::broken_intra_doc_links)]

//@ has foo/demo/index.html
//@ has - '//a[@href="../struct.Foo.html"]' 'Foo'
//@ has - '//a[@href="struct.DemoStruct.html"]' 'DemoStruct'

/// Outer doc-comment [`Foo`].
pub mod demo {
    //!
    //! Inner doc-comment with link: [`DemoStruct`]

    pub struct DemoStruct;
}

pub struct Foo;
