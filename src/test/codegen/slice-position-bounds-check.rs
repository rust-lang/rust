// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-system-llvm
// compile-flags: -O
#![crate_type = "lib"]

fn search<T: Ord + Eq>(arr: &mut [T], a: &T) -> Result<usize, ()> {
    match arr.iter().position(|x| x == a) {
        Some(p) => {
            Ok(p)
        },
        None => Err(()),
    }
}

// CHECK-LABEL: @position_no_bounds_check
#[no_mangle]
pub fn position_no_bounds_check(y: &mut [u32], x: &u32, z: &u32) -> bool {
    // This contains "call assume" so we cannot just rule out all calls
    // CHECK-NOT: panic
    if let Ok(p) = search(y, x) {
      y[p] == *z
    } else {
      false
    }
}

// just to make sure that panicking really emits "panic" somewhere in the IR
// CHECK-LABEL: @test_check
#[no_mangle]
pub fn test_check() {
    // CHECK: panic
    unreachable!()
}
