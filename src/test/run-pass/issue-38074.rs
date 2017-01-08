// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(platform_intrinsics, repr_simd)]

extern "platform-intrinsic" {
    fn simd_shuffle2<T, U>(x: T, y: T, idx: [u32; 2]) -> U;
}

#[repr(simd)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
struct u64x2(u64, u64);

fn main() {
    let a = u64x2(1, 2);
    let r: u64x2 = unsafe { simd_shuffle2(a, a, [0-0, 0-0]) };
    assert_eq!(r.0, 1);
    assert_eq!(r.1, 1);
}
