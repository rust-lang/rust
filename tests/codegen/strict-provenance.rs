// compile-flags: -O

#![crate_type = "lib"]
#![feature(strict_provenance)]

use std::ptr;

// CHECK-LABEL: @invalid
#[no_mangle]
fn invalid(addr: usize) -> *const () {
    // CHECK: start
    // CHECK-NEXT: %0 = getelementptr i8, ptr null, i64 %addr
    // CHECK-NEXT: ret ptr %0
    ptr::invalid(addr)
}
