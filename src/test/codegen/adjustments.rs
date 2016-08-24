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

// CHECK-LABEL: @no_op_slice_adjustment
#[no_mangle]
pub fn no_op_slice_adjustment(x: &[u8]) -> &[u8] {
    // We used to generate an extra alloca and memcpy for the block's trailing expression value, so
    // check that we copy directly to the return value slot
// CHECK: %2 = insertvalue { i8*, [[USIZE]] } undef, i8* %0, 0
// CHECK: %3 = insertvalue { i8*, [[USIZE]] } %2, [[USIZE]] %1, 1
// CHECK: ret { i8*, [[USIZE]] } %3
    { x }
}

// CHECK-LABEL: @no_op_slice_adjustment2
#[no_mangle]
pub fn no_op_slice_adjustment2(x: &[u8]) -> &[u8] {
    // We used to generate an extra alloca and memcpy for the function's return value, so check
    // that there's no memcpy (the slice is written to sret_slot element-wise)
// CHECK-NOT: call void @llvm.memcpy.
    no_op_slice_adjustment(x)
}
