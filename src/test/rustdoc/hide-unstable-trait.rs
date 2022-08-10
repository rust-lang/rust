// aux-build:unstable-trait.rs

#![crate_name = "foo"]
#![feature(private_trait)]

extern crate unstable_trait;

// @hastext foo/struct.Foo.html 'bar'
// @hastext foo/struct.Foo.html 'bar2'
#[doc(inline)]
pub use unstable_trait::Foo;
