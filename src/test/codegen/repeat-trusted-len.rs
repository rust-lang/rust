// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -O
// ignore-tidy-linelength

#![crate_type = "lib"]

use std::iter;

// CHECK: @helper([[USIZE:i[0-9]+]] %arg0)
#[no_mangle]
pub fn helper(_: usize) {
}

// CHECK-LABEL: @repeat_take_collect
#[no_mangle]
pub fn repeat_take_collect() -> Vec<u8> {
// CHECK: call void @llvm.memset.p0i8.[[USIZE]](i8* {{(nonnull )?}}%{{[0-9]+}}, i8 42, [[USIZE]] 100000, i32 1, i1 false)
    iter::repeat(42).take(100000).collect()
}
