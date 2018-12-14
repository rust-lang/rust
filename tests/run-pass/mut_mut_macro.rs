// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(clippy::mut_mut, clippy::zero_ptr, clippy::cmp_nan)]
#![allow(dead_code)]

// compiletest + extern crates doesn't work together
//#[macro_use]
//extern crate lazy_static;

//use std::collections::HashMap;

// ensure that we don't suggest `is_nan` and `is_null` inside constants
// FIXME: once const fn is stable, suggest these functions again in constants
const BAA: *const i32 = 0 as *const i32;
static mut BAR: *const i32 = BAA;
static mut FOO: *const i32 = 0 as *const i32;
static mut BUH: bool = 42.0 < std::f32::NAN;

#[allow(unused_variables, unused_mut)]
fn main() {
    /*
    lazy_static! {
        static ref MUT_MAP : HashMap<usize, &'static str> = {
            let mut m = HashMap::new();
            m.insert(0, "zero");
            m
        };
        static ref MUT_COUNT : usize = MUT_MAP.len();
    }
    assert_eq!(*MUT_COUNT, 1);
    */
    // FIXME: don't lint in array length, requires `check_body`
    //let _ = [""; (42.0 < std::f32::NAN) as usize];
}
