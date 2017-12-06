// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
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

struct Foo<T>(T);

impl Foo<&'s u8> {
    fn bar<'s>(&self, x: &'s u8) {} //~ ERROR shadows a lifetime name
    fn baz(x: for<'s> fn(&'s u32)) {} //~ ERROR shadows a lifetime name
}

fn main() {}
