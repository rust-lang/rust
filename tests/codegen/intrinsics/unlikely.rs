//@ compile-flags: -O
#![crate_type = "lib"]
#![feature(core_intrinsics)]

#[cold]
fn cold_path() {}

#[inline(always)]
fn unlikely(x: bool) -> bool {
    if x {
        cold_path();
        true
    } else {
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
    if unlikely(x) {
        path_a();
    } else {
        path_b();
    }
}

// CHECK-LABEL: @f(
// CHECK: br i1 %x, label %bb2, label %bb4, !prof ![[NUM:[0-9]+]]
// CHECK: bb4:
// CHECK: path_b
// CHECK: bb2:
// CHECK-NOT: cold_path
// CHECK: path_a
// CHECK: ![[NUM]] = !{!"branch_weights",{{.*,}} i32 1, i32 2000}
