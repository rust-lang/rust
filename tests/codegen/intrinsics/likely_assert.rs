//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

#[no_mangle]
pub fn test_assert(x: bool) {
    assert!(x);
}

// check that assert! emits branch weights

// CHECK-LABEL: @test_assert(
// CHECK: br i1 %x, label %bb2, label %bb1, !prof ![[NUM:[0-9]+]]
// CHECK: bb1:
// CHECK: panic
// CHECK: bb2:
// CHECK: ret void
// CHECK: ![[NUM]] = !{!"branch_weights", {{(!"expected", )?}}i32 2000, i32 1}
