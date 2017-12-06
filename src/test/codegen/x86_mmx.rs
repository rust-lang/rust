// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-arm
// ignore-aarch64
// ignore-emscripten
// compile-flags: -O

#![feature(repr_simd)]
#![crate_type="lib"]

#[repr(simd)]
#[derive(Clone, Copy)]
pub struct i8x8(u64);

#[no_mangle]
pub fn a(a: &mut i8x8, b: i8x8) -> i8x8 {
    // CHECK-LABEL: define x86_mmx @a(x86_mmx*{{.*}}, x86_mmx{{.*}})
    // CHECK: store x86_mmx %b, x86_mmx* %a
    // CHECK: ret x86_mmx %b
    *a = b;
    return b
}
