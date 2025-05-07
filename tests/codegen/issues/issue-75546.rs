//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

// Test that LLVM can eliminate the impossible `i == 0` check.

// CHECK-LABEL: @issue_75546
#[no_mangle]
pub fn issue_75546() {
    let mut i = 1u32;
    while i < u32::MAX {
        // CHECK-NOT: panic
        if i == 0 {
            panic!();
        }
        i += 1;
    }
}
