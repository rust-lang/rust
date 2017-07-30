// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// no-system-llvm
// compile-flags: -O
#![crate_type="lib"]

#[no_mangle]
pub fn sum_me() -> i32 {
    // CHECK-LABEL: @sum_me
    // CHECK-NEXT: {{^.*:$}}
    // CHECK-NEXT: ret i32 6
    vec![1, 2, 3].iter().sum::<i32>()
}
