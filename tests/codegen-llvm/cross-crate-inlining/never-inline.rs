//@ compile-flags: -Copt-level=3
//@ aux-build:never.rs

#![crate_type = "lib"]

extern crate never;

// Check that we do not inline a cross-crate call, even though it is a leaf
#[no_mangle]
pub fn outer() -> String {
    // CHECK: call {{.*}}leaf_fn
    never::leaf_fn()
}
