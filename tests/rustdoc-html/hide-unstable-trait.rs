//@ aux-build:unstable-trait.rs

#![crate_name = "foo"]
#![feature(private_trait)]

extern crate unstable_trait;

//@ hasraw foo/type.Foo.html 'bar'
//@ hasraw foo/type.Foo.html 'bar2'
#[doc(inline)]
pub use unstable_trait::Foo;
