//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]
#![feature(cold_path)]

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
pub fn test2(x: i32) {
    match x > 0 {
        true => path_a(),
        false => {
            cold_path();
            path_b()
        }
    }

    // CHECK-LABEL: @test2(
    // CHECK: br i1 %_2, label %bb2, label %bb1, !prof ![[NUM]]
    // CHECK: bb1:
    // CHECK: path_b
    // CHECK: bb2:
    // CHECK: path_a
}

// CHECK: ![[NUM]] = !{!"branch_weights", {{(!"expected", )?}}i32 2000, i32 1}
