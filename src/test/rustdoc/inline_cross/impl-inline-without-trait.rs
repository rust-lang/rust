// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:impl-inline-without-trait.rs
// build-aux-docs
// ignore-cross-compile

#![crate_name = "foo"]

extern crate impl_inline_without_trait;

// @has 'foo/struct.MyStruct.html'
// @has - '//*[@id="method.my_trait_method"]' 'fn my_trait_method()'
// @has - '//*[@class="docblock"]' 'docs for my_trait_method'
pub use impl_inline_without_trait::MyStruct;
