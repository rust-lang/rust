// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

#![feature(intrinsics)]

extern "rust-intrinsic" {
    #[cfg(stage0)]
    fn sqrtf32(x: f32) -> f32;
    #[cfg(not(stage0))]
    fn sqrt<T>(x: T) -> T;
}
// CHECK: @llvm.sqrt.f32(float) #{{[0-9]*}}

#[cfg(stage0)]
fn main() {
    unsafe { sqrtf32(0.0f32); }
}

#[cfg(not(stage0))]
fn main() {
    unsafe { sqrtf(0.0f32); }
}
