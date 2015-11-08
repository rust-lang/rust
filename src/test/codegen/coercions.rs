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

static X: i32 = 5;

// CHECK-LABEL: @raw_ptr_to_raw_ptr_noop
// CHECK-NOT: alloca
#[no_mangle]
pub fn raw_ptr_to_raw_ptr_noop() -> *const i32{
    &X as *const i32
}

// CHECK-LABEL: @reference_to_raw_ptr_noop
// CHECK-NOT: alloca
#[no_mangle]
pub fn reference_to_raw_ptr_noop() -> *const i32 {
    &X
}
