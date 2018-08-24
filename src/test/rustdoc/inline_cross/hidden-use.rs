// aux-build:rustdoc-hidden.rs
// build-aux-docs
// ignore-cross-compile

extern crate rustdoc_hidden;

// @has hidden_use/index.html
// @!has - 'rustdoc_hidden'
// @!has - 'Bar'
// @!has hidden_use/struct.Bar.html
#[doc(hidden)]
pub use rustdoc_hidden::Bar;
