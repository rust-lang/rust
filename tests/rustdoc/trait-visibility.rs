//@ aux-build:trait-visibility.rs

#![crate_name = "foo"]

extern crate trait_visibility;

//@ has foo/trait.Bar.html '//a[@href="#tymethod.foo"]/..' "fn foo()"
pub use trait_visibility::Bar;
