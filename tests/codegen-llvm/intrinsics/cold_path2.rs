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

#[no_mangle]
pub fn test(x: Option<bool>) {
    if let Some(_) = x {
        path_a();
    } else {
        cold_path();
        path_b();
    }

    // CHECK-LABEL: void @test(i8{{.+}}%x)
    // CHECK: %[[IS_NONE:.+]] = icmp eq i8 %x, 2
    // CHECK: br i1 %[[IS_NONE]], label %[[TRUE:bb[0-9]+]], label %[[FALSE:bb[0-9]+]], !prof ![[NUM:[0-9]+]]
    // CHECK: [[FALSE]]:
    // CHECK: path_a
    // CHECK: [[TRUE]]:
    // CHECK: path_b
}

// CHECK: ![[NUM]] = !{!"branch_weights", {{(!"expected", )?}}i32 1, i32 2000}
