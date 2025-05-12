//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]
#![feature(likely_unlikely)]

use std::hint::unlikely;

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
    if unlikely(x) {
        path_a();
    } else {
        path_b();
    }

    // CHECK-LABEL: @test1(
    // CHECK: br i1 %x, label %bb2, label %bb4, !prof ![[NUM:[0-9]+]]
    // CHECK: bb4:
    // CHECK: path_b
    // CHECK: bb2:
    // CHECK: path_a
}

#[no_mangle]
pub fn test2(x: i32) {
    match unlikely(x > 0) {
        true => path_a(),
        false => path_b(),
    }

    // CHECK-LABEL: @test2(
    // CHECK: br i1 %_2, label %bb2, label %bb4, !prof ![[NUM]]
    // CHECK: bb4:
    // CHECK: path_b
    // CHECK: bb2:
    // CHECK: path_a
}

#[no_mangle]
pub fn test3(x: i8) {
    match unlikely(x < 7) {
        true => path_a(),
        _ => path_b(),
    }

    // CHECK-LABEL: @test3(
    // CHECK: br i1 %_2, label %bb2, label %bb4, !prof ![[NUM]]
    // CHECK: bb4:
    // CHECK: path_b
    // CHECK: bb2:
    // CHECK: path_a
}

#[no_mangle]
pub fn test4(x: u64) {
    match unlikely(x != 33) {
        false => path_a(),
        _ => path_b(),
    }

    // CHECK-LABEL: @test4(
    // CHECK: br i1 %0, label %bb4, label %bb2, !prof ![[NUM2:[0-9]+]]
    // CHECK: bb4:
    // CHECK: path_a
    // CHECK: bb2:
    // CHECK: path_b
}

// CHECK: ![[NUM]] = !{!"branch_weights", {{(!"expected", )?}}i32 1, i32 2000}
