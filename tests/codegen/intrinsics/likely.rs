//@ compile-flags: -O
#![crate_type = "lib"]
#![feature(core_intrinsics)]

#[cold]
fn cold_path() {}

#[inline(always)]
fn likely(x: bool) -> bool {
    if x {
        true
    } else {
        cold_path();
        false
    }
}

#[inline(never)]
#[no_mangle]
pub fn path_a() {
    println!("fast_path");
}

#[inline(never)]
#[no_mangle]
pub fn path_b() {
    println!("slow_path");
}

#[no_mangle]
pub fn f(x: bool) {
    if likely(x) {
        path_a();
    } else {
        path_b();
    }
}

// CHECK-LABEL: @f(
// CHECK: br i1 %x, label %bb2, label %bb3, !prof ![[NUM:[0-9]+]]
// CHECK: bb3:
// CHECK: path_b
// CHECK: bb2:
// CHECK: path_a
// CHECK: ![[NUM]] = !{!"branch_weights", i32 2000, i32 1}
