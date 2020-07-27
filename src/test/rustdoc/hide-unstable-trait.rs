// aux-build:unstable-trait.rs

#![crate_name = "foo"]
#![feature(private_trait)]

extern crate unstable_trait;

// @has foo/struct.Foo.html 'bar'
// @has foo/struct.Foo.html 'bar2'
#[doc(inline)]
pub use unstable_trait::Foo;
