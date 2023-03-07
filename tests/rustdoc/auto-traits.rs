// aux-build:auto-traits.rs

#![feature(auto_traits)]

#![crate_name = "foo"]

extern crate auto_traits;

// @has 'foo/trait.Foo.html' '//pre' 'pub unsafe auto trait Foo'
pub unsafe auto trait Foo {}

// @has 'foo/trait.Bar.html' '//pre' 'pub unsafe auto trait Bar'
pub use auto_traits::Bar;
