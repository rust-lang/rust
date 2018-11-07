// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_transmute)]
#![allow(const_err)] // make sure we cannot allow away the errors tested here

use std::mem;

#[derive(Copy, Clone)]
enum Bar {}

const BAD_BAD_BAD: Bar = unsafe { mem::transmute(()) };
//~^ ERROR it is undefined behavior to use this value

const BAD_BAD_REF: &Bar = unsafe { mem::transmute(1usize) };
//~^ ERROR it is undefined behavior to use this value

const BAD_BAD_ARRAY: [Bar; 1] = unsafe { mem::transmute(()) };
//~^ ERROR it is undefined behavior to use this value

fn main() {
}
