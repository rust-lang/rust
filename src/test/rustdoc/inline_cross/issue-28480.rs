// aux-build:rustdoc-hidden-sig.rs
// build-aux-docs
// ignore-cross-compile

// @has rustdoc_hidden_sig/struct.Bar.html
// @!has -  '//a/@title' 'Hidden'
// @has -  '//a' 'u8'
extern crate rustdoc_hidden_sig;

// @has issue_28480/struct.Bar.html
// @!has -  '//a/@title' 'Hidden'
// @has -  '//a' 'u8'
pub use rustdoc_hidden_sig::Bar;
