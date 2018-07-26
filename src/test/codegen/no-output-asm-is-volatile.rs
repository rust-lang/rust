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

// ignore-asmjs

#![feature(asm)]
#![crate_type = "lib"]

// Check that inline assembly expressions without any outputs
// are marked as having side effects / being volatile

// CHECK-LABEL: @assembly
#[no_mangle]
pub fn assembly() {
    unsafe { asm!("") }
// CHECK: tail call void asm sideeffect "", {{.*}}
}
