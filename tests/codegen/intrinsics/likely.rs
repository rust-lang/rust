//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::likely;

#[inline(never)]
#[no_mangle]
pub fn path_a() {
    println!("path a");
}

#[inline(never)]
#[no_mangle]
pub fn path_b() {
    println!("path b");
}

#[no_mangle]
pub fn test_likely(x: bool) {
    if likely(x) {
        path_a();
    } else {
        path_b();
    }
}

// CHECK-LABEL: @test_likely(
// CHECK: br i1 %x, label %bb3, label %bb1, !prof ![[NUM:[0-9]+]]
// CHECK: bb1:
// CHECK-NOT: cold_path
// CHECK: path_b
// CHECK: bb3:
// CHECK: path_a
// CHECK: ![[NUM]] = !{!"branch_weights", {{(!"expected", )?}}i32 2000, i32 1}
