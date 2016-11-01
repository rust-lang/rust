// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes -O

#![crate_type = "lib"]

// CHECK-LABEL: @zip_copy
#[no_mangle]
pub fn zip_copy(xs: &[u8], ys: &mut [u8]) {
// CHECK: memcpy
    for (x, y) in xs.iter().zip(ys) {
        *y = *x;
    }
}

// CHECK-LABEL: @zip_copy_mapped
#[no_mangle]
pub fn zip_copy_mapped(xs: &[u8], ys: &mut [u8]) {
// CHECK: memcpy
    for (x, y) in xs.iter().map(|&x| x).zip(ys) {
        *y = x;
    }
}
