// aux-build:masked.rs

#![feature(doc_masked)]

#![crate_name = "foo"]

#[doc(masked)]
extern crate masked;

// @!has 'search-index.js' 'masked_method'

// @!has 'foo/struct.String.html' 'MaskedTrait'
// @!has 'foo/struct.String.html' 'masked_method'
pub use std::string::String;

// @!has 'foo/trait.Clone.html' 'MaskedStruct'
pub use std::clone::Clone;

// @!has 'foo/struct.MyStruct.html' 'MaskedTrait'
// @!has 'foo/struct.MyStruct.html' 'masked_method'
pub struct MyStruct;

impl masked::MaskedTrait for MyStruct {
    fn masked_method() {}
}

// @!has 'foo/trait.MyTrait.html' 'MaskedStruct'
pub trait MyTrait {}

impl MyTrait for masked::MaskedStruct {}
