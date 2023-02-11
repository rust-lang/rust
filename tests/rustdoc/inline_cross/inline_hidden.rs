// aux-build:rustdoc-hidden.rs
// build-aux-docs
// ignore-cross-compile

extern crate rustdoc_hidden;

#[doc(no_inline)]
pub use rustdoc_hidden::Foo;

// @has inline_hidden/fn.foo.html
// @!has - '//a/@title' 'Foo'
pub fn foo(_: Foo) {}
