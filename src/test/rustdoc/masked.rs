// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
