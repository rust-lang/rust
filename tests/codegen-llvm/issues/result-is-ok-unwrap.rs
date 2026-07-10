// Checking `Result::is_ok()` should make a following `unwrap()` branch-free.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::hint::black_box;

// CHECK-LABEL: @unwrap_after_is_ok
#[no_mangle]
pub fn unwrap_after_is_ok(arg: Result<u64, u32>) {
    // CHECK-NOT: unwrap_failed
    // CHECK-NOT: panic
    if arg.is_ok() {
        let value = arg.unwrap();
        if value == 42 {
            black_box(value);
        }
    }
}
