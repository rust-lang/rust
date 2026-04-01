//@ compile-flags: -Copt-level=3
//@ aux-build:always.rs

#![crate_type = "lib"]

extern crate always;

// Check that we inline a cross-crate call, even though it isn't a leaf
#[no_mangle]
pub fn outer() -> String {
    // CHECK-NOT: call {{.*}}stem_fn
    always::stem_fn()
}
