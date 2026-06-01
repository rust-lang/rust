//@ compile-flags: -Copt-level=3
//@ only-x86_64
#![crate_type = "lib"]

// This test checks that hint::cold_path still works in #[target_feature] functions.

use std::hint::cold_path;

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
pub fn test1(x: bool) {
    if x {
        path_a();
    } else {
        cold_path();
        path_b();
    }

    // CHECK-LABEL: @test1(
    // CHECK: br i1 %x, label %bb1, label %bb2, !prof ![[NUM:[0-9]+]]
    // CHECK: bb2:
    // CHECK: path_b
    // CHECK: bb1:
    // CHECK: path_a
}

#[no_mangle]
#[target_feature(enable = "sse2")]
pub fn with_target_feature(x: bool) {
    if x {
        path_a();
    } else {
        cold_path();
        path_b();
    }

    // CHECK-LABEL: @with_target_feature(
    // CHECK: br i1 %x, label %bb1, label %bb2, !prof ![[NUM]]
    // CHECK: bb2:
    // CHECK: path_b
    // CHECK: bb1:
    // CHECK: path_a
}

// CHECK: ![[NUM]] = !{!"branch_weights", {{(!"expected", )?}}i32 2000, i32 1}
