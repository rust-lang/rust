//@ compile-flags: -Copt-level=3
#![feature(panic_internals, const_eval_select, rustc_allow_const_fn_unstable, core_intrinsics)]
#![crate_type = "lib"]

// check that assert! and const_assert! emit branch weights

#[no_mangle]
pub fn test_assert(x: bool) {
    assert!(x);
}

// CHECK-LABEL: @test_assert(
// CHECK: br i1 %x, label %bb2, label %bb1, !prof ![[NUM:[0-9]+]]
// CHECK: bb1:
// CHECK: panic
// CHECK: bb2:
// CHECK: ret void

#[no_mangle]
pub fn test_const_assert(x: bool) {
    core::panic::const_assert!(x, "", "",);
}

// CHECK-LABEL: @test_const_assert(
// CHECK: br i1 %x, label %bb2, label %bb1, !prof ![[NUM:[0-9]+]]
// CHECK: bb1:
// CHECK: panic
// CHECK: bb2:
// CHECK: ret void

// CHECK: ![[NUM]] = !{!"branch_weights", {{(!"expected", )?}}i32 2000, i32 1}
