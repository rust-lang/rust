// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -O

#![feature(core_intrinsics)]
#![crate_type = "lib"]

#[no_mangle]
pub fn a(a: &mut u32, b: u32) {
    // CHECK-LABEL: define void @a
    // CHECK: store i32 %b, i32* %a, align 4, !nontemporal
    unsafe {
        std::intrinsics::nontemporal_store(a, b);
    }
}
