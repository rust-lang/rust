// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the use of smid types in the ffi is gated by `smid_ffi` feature gate.

#![feature(simd)]

#[repr(C)]
#[derive(Copy, Clone)]
#[simd]
pub struct f32x4(f32, f32, f32, f32);

#[allow(dead_code)]
extern {
    fn foo(x: f32x4);
    //~^ ERROR use of SIMD type `f32x4` in FFI is highly experimental and may result in invalid code
    //~| HELP add #![feature(simd_ffi)] to the crate attributes to enable
}

fn main() {}
