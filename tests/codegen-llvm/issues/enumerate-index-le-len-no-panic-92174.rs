//! Regression test for <https://github.com/rust-lang/rust/issues/92174>.
//!
//! The index from `enumerate()` over a slice is never greater than the slice
//! length, so the assertion and its panic must be removed.

//@ compile-flags: -Copt-level=3
//@ ignore-wasm32 LLVM does not remove the assertion on wasm32

#![crate_type = "lib"]

// CHECK-LABEL: @enumerate_index_le_len
#[no_mangle]
pub fn enumerate_index_le_len(v: &[u32]) {
    // CHECK-NOT: panic
    // CHECK-NOT: br {{.*}}
    // CHECK: ret void
    for (i, _) in v.iter().enumerate() {
        assert!(i <= v.len());
    }
}
