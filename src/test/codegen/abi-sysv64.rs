// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks if the correct annotation for the sysv64 ABI is passed to
// llvm. Also checks that the abi-sysv64 feature gate allows usage
// of the sysv64 abi.

// ignore-arm
// ignore-aarch64

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(abi_sysv64)]

// CHECK: define x86_64_sysvcc i64 @has_sysv64_abi
#[no_mangle]
pub extern "sysv64" fn has_sysv64_abi(a: i64) -> i64 {
    a * 2
}
