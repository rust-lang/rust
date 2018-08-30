// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten FIXME(#45351)

#![feature(repr_simd, platform_intrinsics)]

#[repr(C)]
#[repr(simd)]
#[derive(Copy, Clone, Debug)]
pub struct char3(pub i8, pub i8, pub i8);

#[repr(C)]
#[repr(simd)]
#[derive(Copy, Clone, Debug)]
pub struct short3(pub i16, pub i16, pub i16);

extern "platform-intrinsic" {
    fn simd_cast<T, U>(x: T) -> U;
}

fn main() {
    let cast: short3 = unsafe { simd_cast(char3(10, -3, -9)) };

    println!("{:?}", cast);
}
