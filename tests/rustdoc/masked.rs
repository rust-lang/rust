//@ aux-build:masked.rs

#![feature(doc_masked)]

#![crate_name = "foo"]

#[doc(masked)]
extern crate masked;

//@ !hasraw 'search.index/name/*.js' 'masked_method'

//@ !hasraw 'foo/struct.String.html' 'MaskedTrait'
//@ !hasraw 'foo/struct.String.html' 'MaskedBlanketTrait'
//@ !hasraw 'foo/struct.String.html' 'masked_method'
pub use std::string::String;

//@ !hasraw 'foo/trait.Clone.html' 'MaskedStruct'
pub use std::clone::Clone;

//@ !hasraw 'foo/struct.MyStruct.html' 'MaskedTrait'
//@ !hasraw 'foo/struct.MyStruct.html' 'masked_method'
pub struct MyStruct;

impl masked::MaskedTrait for MyStruct {
    fn masked_method() {}
}

//@ !hasraw 'foo/trait.MyTrait.html' 'MaskedStruct'
pub trait MyTrait {}

impl MyTrait for masked::MaskedStruct {}
