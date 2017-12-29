// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(repr_simd)]
#![feature(platform_intrinsics)]

#[repr(simd)]
struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);

extern "platform-intrinsic" {
    fn x86_mm_adds_ep16(x: i16x8, y: i16x8) -> i16x8; //~ ERROR E0441
}

fn main() {}
