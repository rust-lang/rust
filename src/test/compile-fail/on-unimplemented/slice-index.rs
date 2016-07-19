// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test new Index error message for slices
// ignore-tidy-linelength

#![feature(rustc_attrs)]

use std::ops::Index;

#[rustc_error]
fn main() {
    let x = &[1, 2, 3] as &[i32];
    x[1i32]; //~ ERROR E0277
             //~| NOTE slice indices are of type `usize` or ranges of `usize`
             //~| NOTE trait `std::slice::SliceIndex<i32>` is not implemented for `i32`
             //~| NOTE required because of the requirements on the impl of `std::ops::Index<i32>`
    x[..1i32]; //~ ERROR E0277
               //~| NOTE slice indices are of type `usize` or ranges of `usize`
               //~| NOTE trait `std::slice::SliceIndex<i32>` is not implemented for `std::ops::RangeTo<i32>`
               //~| NOTE requirements on the impl of `std::ops::Index<std::ops::RangeTo<i32>>`
}
