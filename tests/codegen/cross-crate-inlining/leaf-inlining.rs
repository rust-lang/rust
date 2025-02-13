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
