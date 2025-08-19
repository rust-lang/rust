//@ compile-flags: -Z normalize-docs --document-private-items -Zunstable-options --show-type-layout
//@ aux-build:wrap-unnamable-type.rs
//@ build-aux-docs

// regression test for https://github.com/rust-lang/rust/issues/143222
// makes sure normalizing docs does not cause us to link to unnamable types
// in cross-crate reexports.

#![crate_name = "foo"]

extern crate wrap_unnamable_type as helper;

//@ has 'foo/struct.Foo.html'
//@ !hasraw - 'struct.Bar.html'
#[doc(inline)]
pub use helper::Foo;
