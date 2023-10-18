// compile-flags: -O -Zcross-crate-inline-threshold=yes
// aux-build:leaf.rs

#![crate_type = "lib"]

extern crate leaf;

// Check that we inline a leaf cross-crate call
#[no_mangle]
pub fn leaf_outer() -> String {
    // CHECK-NOT: call {{.*}}leaf_fn
    leaf::leaf_fn()
}

// Check that we inline a cross-crate call where the callee contains a single call
#[no_mangle]
pub fn stem_outer() -> String {
    // CHECK-NOT: call {{.*}}stem_fn
    leaf::stem_fn()
}

// Check that we do not inline a cross-crate call where the callee contains multiple calls
#[no_mangle]
pub fn multi_stem_outer() -> String {
    // CHECK: call {{.*}}multi_stem_fn
    leaf::multi_stem_fn()
}
