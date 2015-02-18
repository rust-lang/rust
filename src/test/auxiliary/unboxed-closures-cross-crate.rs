// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]

use std::ops::Add;

#[inline]
pub fn has_closures() -> uint {
    let x = 1_usize;
    let mut f = move || x;
    let y = 1_usize;
    let g = || y;
    f() + g()
}

pub fn has_generic_closures<T: Add<Output=T> + Copy>(x: T, y: T) -> T {
    let mut f = move || x;
    let g = || y;
    f() + g()
}
