// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// min-llvm-version 4.0
// compile-flags: -O
// ignore-x86
// ignore-arm
// ignore-emscripten
// ignore-gnux32
// ignore 32-bit platforms (LLVM has a bug with them)

// See issue #37945.

#![crate_type = "lib"]

use std::slice::Iter;

// CHECK-LABEL: @is_empty_1
#[no_mangle]
pub fn is_empty_1(xs: Iter<f32>) -> bool {
// CHECK-NOT: icmp eq float* {{.*}}, null
    {xs}.next().is_none()
}

// CHECK-LABEL: @is_empty_2
#[no_mangle]
pub fn is_empty_2(xs: Iter<f32>) -> bool {
// CHECK-NOT: icmp eq float* {{.*}}, null
    xs.map(|&x| x).next().is_none()
}
