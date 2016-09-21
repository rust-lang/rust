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

#![crate_type = "lib"]

// Hack to get the correct size for the length part in slices
// CHECK: @helper([[USIZE:i[0-9]+]])
#[no_mangle]
fn helper(_: usize) {
}

// CHECK-LABEL: @ref_dst
#[no_mangle]
pub fn ref_dst(s: &[u8]) {
    // We used to generate an extra alloca and memcpy to ref the dst, so check that we copy
    // directly to the alloca for "x"
// CHECK: [[X0:%[0-9]+]] = getelementptr {{.*}} { i8*, [[USIZE]] }* %x, i32 0, i32 0
// CHECK: store i8* %0, i8** [[X0]]
// CHECK: [[X1:%[0-9]+]] = getelementptr {{.*}} { i8*, [[USIZE]] }* %x, i32 0, i32 1
// CHECK: store [[USIZE]] %1, [[USIZE]]* [[X1]]

    let x = &*s;
    &x; // keep variable in an alloca
}
