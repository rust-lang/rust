// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C no-prepopulate-passes

#![feature(core_intrinsics)]
#![crate_type = "lib"]

// test that `move_val_init` actually avoids big allocas

use std::intrinsics::move_val_init;

pub struct Big {
    pub data: [u8; 65536]
}

// CHECK-LABEL: @test_mvi
#[no_mangle]
pub unsafe fn test_mvi(target: *mut Big, make_big: fn() -> Big) {
    // CHECK: call void %make_big(%Big*{{[^%]*}} %target)
    move_val_init(target, make_big());
}
