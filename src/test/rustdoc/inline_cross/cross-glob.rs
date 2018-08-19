// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:cross-glob.rs
// build-aux-docs
// ignore-cross-compile

extern crate inner;

// @has cross_glob/struct.SomeStruct.html
// @has cross_glob/fn.some_fn.html
// @!has cross_glob/index.html '//code' 'pub use inner::*;'
#[doc(inline)]
pub use inner::*;
