//@ compile-flags: -Z normalize-docs --document-private-items -Zunstable-options --show-type-layout
//@ aux-build:wrap-unnamable-type.rs
//@ build-aux-docs

#![crate_name = "foo"]

extern crate wrap_unnamable_type as helper;
//extern crate helper;
//@ has 'foo/struct.Foo.html'
//@ !hasraw - '_/struct.Bar.html'
#[doc(inline)]
pub use helper::Foo;
