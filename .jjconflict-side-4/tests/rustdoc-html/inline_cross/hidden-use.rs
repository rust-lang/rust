//@ aux-build:rustdoc-hidden.rs
//@ build-aux-docs
//@ ignore-cross-compile

extern crate rustdoc_hidden;

//@ has hidden_use/index.html
//@ !hasraw - 'rustdoc_hidden'
//@ !hasraw - 'Bar'
//@ !has hidden_use/struct.Bar.html
#[doc(hidden)]
pub use rustdoc_hidden::Bar;
