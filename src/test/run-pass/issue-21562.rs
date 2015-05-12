// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core)]

extern crate core;
use core::marker::Sync;

static SARRAY: [i32; 1] = [11];

struct MyStruct {
    pub arr: *const [i32],
}
unsafe impl Sync for MyStruct {}

static mystruct: MyStruct = MyStruct {
    arr: &SARRAY
};

fn main() {}
