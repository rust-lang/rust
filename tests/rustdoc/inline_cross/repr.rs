// Regression test for <https://github.com/rust-lang/rust/issues/110698>.
// This test ensures that the re-exported items still have the `#[repr(...)]` attribute.

// aux-build:repr.rs

#![crate_name = "foo"]

extern crate repr;

// @has 'foo/struct.Foo.html'
// @has - '//*[@class="rust item-decl"]//*[@class="code-attribute"]' '#[repr(C)]'
#[doc(inline)]
pub use repr::Foo;
