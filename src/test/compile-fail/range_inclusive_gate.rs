// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure that #![feature(inclusive_range)] is required.

#![feature(inclusive_range_syntax)]
// #![feature(inclusive_range)]

pub fn main() {
    let _: std::ops::RangeInclusive<_> = { use std::intrinsics; 1 } ... { use std::intrinsics; 2 };
    //~^ ERROR use of unstable library feature 'inclusive_range'
    //~| ERROR core_intrinsics
    //~| ERROR core_intrinsics
}


