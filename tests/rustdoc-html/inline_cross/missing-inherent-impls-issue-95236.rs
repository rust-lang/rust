// Regression test for https://github.com/rust-lang/rust/issues/95236
// Rustdoc should render inherent impls the same in both
// re-exported items and their original crate.

//@ aux-build:assoc-items.rs
//@ build-aux-docs
//@ ignore-cross-compile

#![crate_name = "second"]

extern crate assoc_items;

//@ has second/struct.MyStruct.html '//h4[@class="code-header"]' 'pub const PublicConst: u8 = 123'
//@ has second/struct.MyStruct.html '//h4[@class="code-header"]' 'pub fn public_method()'
//@ count second/struct.MyStruct.html '//*[@id="implementations-list"]//h4[@class="code-header"]' 2
//@ !hasraw second/struct.MyStruct.html 'PrivateConst'
//@ !hasraw second/struct.MyStruct.html 'private_method'
//@ has assoc_items/struct.MyStruct.html '//h4[@class="code-header"]' 'pub const PublicConst: u8 = 123'
//@ has assoc_items/struct.MyStruct.html '//h4[@class="code-header"]' 'pub fn public_method()'
//@ count assoc_items/struct.MyStruct.html '//*[@id="implementations-list"]//h4[@class="code-header"]' 2
//@ !hasraw assoc_items/struct.MyStruct.html 'PrivateConst'
//@ !hasraw assoc_items/struct.MyStruct.html 'private_method'
#[doc(inline)]
pub use assoc_items::*;
