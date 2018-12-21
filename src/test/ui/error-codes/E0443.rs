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
struct i8x8(i8, i8, i8, i8, i8, i8, i8, i8);
#[repr(simd)]
struct i64x8(i64, i64, i64, i64, i64, i64, i64, i64);

extern "platform-intrinsic" {
    fn aarch64_vld2_s8(x: *const i8) -> (i8x8, i64x8);
    //~^ ERROR E0443
}

fn main() {}
