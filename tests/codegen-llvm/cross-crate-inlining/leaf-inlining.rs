//@ compile-flags: -Copt-level=3 -Zcross-crate-inline-threshold=yes
//@ aux-build:leaf.rs

#![crate_type = "lib"]

extern crate leaf;

// Check that we inline a leaf cross-crate call
#[no_mangle]
pub fn leaf_outer() -> String {
    // CHECK-NOT: call {{.*}}leaf_fn
    leaf::leaf_fn()
}

// Check that we do not inline a non-leaf cross-crate call
#[no_mangle]
pub fn stem_outer() -> String {
    // CHECK: call {{.*}}stem_fn
    leaf::stem_fn()
}

// Check that we inline functions that call intrinsics
#[no_mangle]
pub fn leaf_with_intrinsic_outer(a: &[u64; 2], b: &[u64; 2]) -> bool {
    // CHECK-NOT: call {{.*}}leaf_with_intrinsic
    leaf::leaf_with_intrinsic(a, b)
}

// Check that we inline functions with assert terminators
#[no_mangle]
pub fn leaf_with_assert(a: i32, b: i32) -> i32 {
    // CHECK-NOT: call {{.*}}leaf_with_assert
    // CHECK: sdiv i32 %a, %b
    // CHECK-NOT: call {{.*}}leaf_with_assert
    leaf::leaf_with_assert(a, b)
}
