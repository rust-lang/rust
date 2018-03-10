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

#![crate_type="rlib"]

// CHECK-LABEL: @memzero
// CHECK-NOT: store
// CHECK: call void @llvm.memset
// CHECK-NOT: store
#[no_mangle]
pub fn memzero(data: &mut [u8]) {
    for i in 0..data.len() {
        data[i] = 0;
    }
}
