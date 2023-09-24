// aux-build:auto-traits.rs

#![feature(rustc_attrs)]

#![crate_name = "foo"]

extern crate auto_traits;

// FIXME(auto_traits): Don't render the contextual keyword `auto` but a "text in an outline"
// @has 'foo/trait.Foo.html' '//pre' 'pub unsafe auto trait Foo'
#[rustc_auto_trait]
pub unsafe trait Foo {}

// FIXME(auto_traits): Don't render the contextual keyword `auto` but a "text in an outline"
// @has 'foo/trait.Bar.html' '//pre' 'pub unsafe auto trait Bar'
pub use auto_traits::Bar;
