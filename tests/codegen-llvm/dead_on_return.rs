//@ compile-flags: -C opt-level=3
//@ min-llvm-version: 21

#![crate_type = "lib"]
#![allow(unused_assignments, unused_variables)]

// Check that the old string is deallocated, but a new one is not initialized.
#[unsafe(no_mangle)]
pub fn test(mut s: String) {
    // CHECK-LABEL: @test
    // CHECK: __rust_dealloc
    // CHECK-NOT: store
    s = String::new();
}
