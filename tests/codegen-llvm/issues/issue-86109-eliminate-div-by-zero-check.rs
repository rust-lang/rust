//@ compile-flags: -Copt-level=3
//! Test for https://github.com/rust-lang/rust/issues/86109
//! Check LLVM can eliminate the impossible division by zero check by
//! ensuring there is no call (to panic) instruction.
//!
//! This has been fixed since `rustc 1.70.0`.

#![crate_type = "lib"]

type T = i16;

// CHECK-LABEL: @foo
#[no_mangle]
pub fn foo(start: T) -> T {
    // CHECK-NOT: panic
    if start <= 0 {
        return 0;
    }
    let mut count = 0;
    for i in start..10_000 {
        if 752 % i != 0 {
            count += 1;
        }
    }
    count
}
