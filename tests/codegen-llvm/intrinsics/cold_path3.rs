//@ compile-flags: -O
#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::cold_path;

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

#[inline(never)]
#[no_mangle]
pub fn path_c() {
    println!("path c");
}

#[inline(never)]
#[no_mangle]
pub fn path_d() {
    println!("path d");
}

#[no_mangle]
pub fn test(x: Option<u32>) {
    match x {
        Some(0) => path_a(),
        Some(1) => {
            cold_path();
            path_b()
        }
        Some(2) => path_c(),
        Some(3) => {
            cold_path();
            path_d()
        }
        _ => path_a(),
    }

    // CHECK-LABEL: @test(
    // CHECK: switch i32 %1, label %bb1 [
    // CHECK: i32 0, label %bb6
    // CHECK: i32 1, label %bb5
    // CHECK: i32 2, label %bb4
    // CHECK: i32 3, label %bb3
    // CHECK: ], !prof ![[NUM1:[0-9]+]]
}

#[no_mangle]
pub fn test2(x: Option<u32>) {
    match x {
        Some(10) => path_a(),
        Some(11) => {
            cold_path();
            path_b()
        }
        Some(12) => {
            unsafe { core::intrinsics::unreachable() };
            path_c()
        }
        Some(13) => {
            cold_path();
            path_d()
        }
        _ => {
            cold_path();
            path_a()
        }
    }

    // CHECK-LABEL: @test2(
    // CHECK: switch i32 %1, label %bb1 [
    // CHECK: i32 10, label %bb5
    // CHECK: i32 11, label %bb4
    // CHECK: i32 13, label %bb3
    // CHECK: ], !prof ![[NUM2:[0-9]+]]
}

// CHECK: ![[NUM1]] = !{!"branch_weights", i32 2000, i32 2000, i32 1, i32 2000, i32 1}
// CHECK: ![[NUM2]] = !{!"branch_weights", i32 1, i32 2000, i32 1, i32 1}
