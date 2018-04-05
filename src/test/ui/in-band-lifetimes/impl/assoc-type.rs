// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we do not yet support elision in associated types, even
// when there is just one name we could take from the impl header.

#![allow(warnings)]

#![feature(in_band_lifetimes)]

trait MyTrait {
    type Output;
}

impl MyTrait for &i32 {
    type Output = &i32;
    //~^ ERROR missing lifetime specifier
}

impl MyTrait for &u32 {
    type Output = &'_ i32;
    //~^ ERROR missing lifetime specifier
}

// This is what you have to do:
impl MyTrait for &'a f32 {
    type Output = &'a f32;
}

fn main() { }
