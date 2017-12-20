// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z borrowck=mir -Z nll

#![allow(dead_code)]
#![feature(dyn_trait)]

use std::fmt::Debug;

fn bar<'a>(x: &'a u32) -> &'static dyn Debug {
    //~^ ERROR free region `'a` does not outlive free region `'static`
    x
    //~^ WARNING not reporting region error due to -Znll
}

fn main() {}
