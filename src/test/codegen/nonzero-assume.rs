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

#![crate_type = "lib"]

use std::num::NonZeroU64;

// CHECK-LABEL: @assume_nonzero
#[no_mangle]
pub fn assume_nonzero(x: u64, y: NonZeroU64) -> u64 {
    x / y.get()
    // CHECK: icmp ne i64 %y, 0
    // CHECK: @llvm.assume
}
