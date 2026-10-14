// Checking `Result::is_ok()` should make a following `unwrap()` branch-free.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::hint::black_box;

// CHECK-LABEL: @unwrap_after_is_ok
#[no_mangle]
pub fn unwrap_after_is_ok(arg: Result<u64, u32>) {
    // CHECK-NOT: unwrap_failed
    if arg.is_ok() {
        let value = arg.unwrap();
        if value == 42 {
            black_box(value);
        }
    }
}

// Ensure our `NOT` check above could actually ever trigger

// CHECK-LABEL: @unguarded_unwrap_calls_unwrap_failed
#[no_mangle]
pub fn unguarded_unwrap_calls_unwrap_failed(arg: Result<u64, u32>) -> u64 {
    // CHECK: call{{.+}}unwrap_failed
    arg.unwrap()
}

// CHECK: declare{{.+}}unwrap_failed
