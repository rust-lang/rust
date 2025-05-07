//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::unlikely;

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
pub fn test_unlikely(x: bool) {
    if unlikely(x) {
        path_a();
    } else {
        path_b();
    }
}

// CHECK-LABEL: @test_unlikely(
// CHECK: br i1 %x, label %bb2, label %bb4, !prof ![[NUM:[0-9]+]]
// CHECK: bb4:
// CHECK: path_b
// CHECK: bb2:
// CHECK-NOT: cold_path
// CHECK: path_a
// CHECK: ![[NUM]] = !{!"branch_weights", {{(!"expected", )?}}i32 1, i32 2000}
