// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![crate_type="lib"]

#[no_mangle]
pub fn alloc_zeroed_test(size: u8) {
    // CHECK-LABEL: @alloc_zeroed_test
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret void
    let x = vec![0u8; size as usize];
    drop(x);
}

#[no_mangle]
pub fn alloc_test(data: u32) {
    // CHECK-LABEL: @alloc_test
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret void
    let x = Box::new(data);
    drop(x);
}
