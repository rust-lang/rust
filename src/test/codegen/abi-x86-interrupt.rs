// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks if the correct annotation for the x86-interrupt ABI is passed to
// llvm. Also checks that the abi_x86_interrupt feature gate allows usage
// of the x86-interrupt abi.

// ignore-arm
// ignore-aarch64
// min-llvm-version 3.8

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(abi_x86_interrupt)]

// CHECK: define x86_intrcc i64 @has_x86_interrupt_abi
#[no_mangle]
pub extern "x86-interrupt" fn has_x86_interrupt_abi(a: i64) -> i64 {
    a * 2
}
