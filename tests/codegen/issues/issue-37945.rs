//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled
//@ ignore-32bit LLVM has a bug with them

// Check that LLVM understands that `Iter` pointer is not null. Issue #37945.

// There used to be a comparison against `null`, so we check that it's not there
// and that the appropriate parameter metadata is.

#![crate_type = "lib"]

use std::slice::Iter;

#[no_mangle]
pub fn is_empty_1(xs: Iter<f32>) -> bool {
    // CHECK-LABEL: @is_empty_1
    // CHECK-SAME: (ptr noundef nonnull{{.*}}%xs.0, {{i32|i64}}{{.+}}%xs.1)

    // CHECK-NOT: null
    // CHECK-NOT: i32 0
    // CHECK-NOT: i64 0

    { xs }.next().is_none()
}

#[no_mangle]
pub fn is_empty_2(xs: Iter<f32>) -> bool {
    // CHECK-LABEL: @is_empty_2
    // CHECK-SAME: (ptr noundef nonnull{{.*}}%xs.0, {{i32|i64}}{{.+}}%xs.1)

    // CHECK-NOT: null
    // CHECK-NOT: i32 0
    // CHECK-NOT: i64 0

    xs.map(|&x| x).next().is_none()
}
