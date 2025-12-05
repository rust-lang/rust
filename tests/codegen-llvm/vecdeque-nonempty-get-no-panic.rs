// Guards against regression for optimization discussed in issue #80836

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::collections::VecDeque;

// CHECK-LABEL: @front
// CHECK: ret void
#[no_mangle]
pub fn front(v: VecDeque<usize>) {
    if !v.is_empty() {
        v.get(0).unwrap();
    }
}
