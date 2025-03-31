//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]
#![feature(likely_unlikely)]

use std::hint::likely;

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
    if likely(x) {
        path_a();
    } else {
        path_b();
    }

    // CHECK-LABEL: @test1(
    // CHECK: br i1 %x, label %bb3, label %bb1, !prof ![[NUM:[0-9]+]]
    // CHECK: bb1:
    // CHECK: path_b
    // CHECK: bb3:
    // CHECK: path_a
}

#[no_mangle]
pub fn test2(x: i32) {
    match likely(x > 0) {
        true => path_a(),
        false => path_b(),
    }

    // CHECK-LABEL: @test2(
    // CHECK: br i1 %_2, label %bb3, label %bb1, !prof ![[NUM]]
    // CHECK: bb1:
    // CHECK: path_b
    // CHECK: bb3:
    // CHECK: path_a
}

#[no_mangle]
pub fn test3(x: i8) {
    match likely(x < 7) {
        true => path_a(),
        _ => path_b(),
    }

    // CHECK-LABEL: @test3(
    // CHECK: br i1 %_2, label %bb3, label %bb1, !prof ![[NUM]]
    // CHECK: bb1:
    // CHECK: path_b
    // CHECK: bb3:
    // CHECK: path_a
}

#[no_mangle]
pub fn test4(x: u64) {
    match likely(x != 33) {
        false => path_a(),
        _ => path_b(),
    }

    // Note that LLVM seems to flip this one, for some reason

    // CHECK-LABEL: @test4(
    // CHECK: %0 = icmp eq i64 %x, 33
    // CHECK: br i1 %0, label %bb1, label %bb3, !prof ![[NUM2:[0-9]+]]
    // CHECK: bb1:
    // CHECK: path_a
    // CHECK: bb3:
    // CHECK: path_b
}

// CHECK: ![[NUM]] = !{!"branch_weights", {{(!"expected", )?}}i32 2000, i32 1}
// CHECK: ![[NUM2]] = !{!"branch_weights", {{(!"expected", )?}}i32 1, i32 2000}
