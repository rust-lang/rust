// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(nll)]
#![allow(warnings)]

fn foo<'a>(x: &'a (u32,)) -> &'a u32 {
    let v = 22;
    &v
    //~^ ERROR `v` does not live long enough [E0597]
}

fn main() {}
