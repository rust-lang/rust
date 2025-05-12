//@ aux-build:rustdoc-default-impl.rs
//@ ignore-cross-compile

extern crate rustdoc_default_impl as foo;

pub use foo::bar;

pub fn wut<T: bar::Bar>() {
}
