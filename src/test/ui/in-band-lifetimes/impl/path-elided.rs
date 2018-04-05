// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![allow(warnings)]

#![feature(in_band_lifetimes)]

trait MyTrait { }

struct Foo<'a> { x: &'a u32 }

impl MyTrait for Foo {
    //~^ ERROR missing lifetime specifier
}

fn main() {}
