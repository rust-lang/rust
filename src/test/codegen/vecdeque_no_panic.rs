// This test checks that `VecDeque::front[_mut]()` and `VecDeque::back[_mut]()` can't panic.

// compile-flags: -O
// min-llvm-version: 11.0.0

#![crate_type = "lib"]

use std::collections::VecDeque;

// CHECK-LABEL: @dont_panic
#[no_mangle]
pub fn dont_panic(v: &mut VecDeque<usize>) {
    // CHECK-NOT: expect
    // CHECK-NOT: panic
    v.front();
    v.front_mut();
    v.back();
    v.back_mut();
}
