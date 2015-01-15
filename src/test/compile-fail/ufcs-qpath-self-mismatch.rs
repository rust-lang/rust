// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Add;

fn main() {
    <i32 as Add<u32>>::add(1, 2);
    //~^ ERROR the trait `core::ops::Add<u32>` is not implemented for the type `i32`
    <i32 as Add<i32>>::add(1u32, 2);
    //~^ ERROR mismatched types
    <i32 as Add<i32>>::add(1, 2u32);
    //~^ ERROR mismatched types
}

