//@ aux-build:impl-inline-without-trait.rs
//@ build-aux-docs
//@ ignore-cross-compile

#![crate_name = "foo"]

extern crate impl_inline_without_trait;

//@ has 'foo/struct.MyStruct.html'
//@ has - '//*[@id="method.my_trait_method"]' 'fn my_trait_method()'
//@ has - '//div[@class="docblock"]' 'docs for my_trait_method'
pub use impl_inline_without_trait::MyStruct;
