// https://github.com/rust-lang/rust/issues/28480
#![crate_name="foobar"]

//@ aux-build:rustdoc-hidden-sig.rs
//@ build-aux-docs
//@ ignore-cross-compile

//@ has rustdoc_hidden_sig/struct.Bar.html
//@ !has -  '//a/@title' 'Hidden'
//@ has -  '//a' 'u8'
extern crate rustdoc_hidden_sig;

//@ has foobar/struct.Bar.html
//@ !has -  '//a/@title' 'Hidden'
//@ has -  '//a' 'u8'
pub use rustdoc_hidden_sig::Bar;
