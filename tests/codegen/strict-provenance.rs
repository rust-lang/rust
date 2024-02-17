//@ compile-flags: -O

#![crate_type = "lib"]
#![feature(strict_provenance)]

use std::ptr;

// CHECK-LABEL: ptr @without_provenance(
// CHECK-SAME: [[USIZE:i[0-9]+]] noundef %addr)
#[no_mangle]
fn without_provenance(addr: usize) -> *const () {
    // CHECK: start
    // CHECK-NEXT: %0 = getelementptr i8, ptr null, [[USIZE]] %addr
    // CHECK-NEXT: ret ptr %0
    ptr::without_provenance(addr)
}
