// `0..slice.len()` should propagate the same upper bound as
// `slice.iter().enumerate()`, allowing the checked index conversion to be
// removed inside the loop.

//@ compile-flags: -Copt-level=3
//@ only-64bit
//@ ignore-s390x OPT missing on s390x https://github.com/llvm/llvm-project/issues/208712

#![crate_type = "lib"]

use std::convert::TryFrom;
use std::hint::black_box;

// CHECK-LABEL: @score_round_enumerate
#[no_mangle]
pub fn score_round_enumerate(candidates: &[bool]) {
    // The length check itself can still fail.
    // CHECK: call void {{.*}}unwrap_failed
    // But the checked conversion inside the loop should not add another panic path.
    // CHECK-NOT: call void {{.*}}unwrap_failed
    // CHECK: ret void
    u32::try_from(candidates.len()).unwrap();

    for (i, _) in candidates.iter().enumerate() {
        u32::try_from(i).unwrap();
        black_box(42);
    }
}

// CHECK-LABEL: @score_round_range
#[no_mangle]
pub fn score_round_range(candidates: &[bool]) {
    // The length check itself can still fail.
    // CHECK: call void {{.*}}unwrap_failed
    // But the checked conversion inside the loop should not add another panic path.
    // CHECK-NOT: call void {{.*}}unwrap_failed
    // CHECK: ret void
    u32::try_from(candidates.len()).unwrap();

    for i in 0..candidates.len() {
        u32::try_from(i).unwrap();
        black_box(42);
    }
}
