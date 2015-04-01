// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we are able to introduce a negative constraint that
// `MyType: !MyTrait` along with other "fundamental" wrappers.

// aux-build:coherence_copy_like_lib.rs

#![feature(rustc_attrs)]
#![allow(dead_code)]

extern crate coherence_copy_like_lib as lib;

struct MyType { x: i32 }

// These are all legal because they are all fundamental types:

impl lib::MyCopy for MyType { }
impl<'a> lib::MyCopy for &'a MyType { }
impl<'a> lib::MyCopy for &'a Box<MyType> { }
impl lib::MyCopy for Box<MyType> { }
impl lib::MyCopy for lib::MyFundamentalStruct<MyType> { }
impl lib::MyCopy for lib::MyFundamentalStruct<Box<MyType>> { }

#[rustc_error]
fn main() { } //~ ERROR compilation successful
