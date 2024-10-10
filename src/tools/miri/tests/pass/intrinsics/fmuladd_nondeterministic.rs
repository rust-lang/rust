#![feature(core_intrinsics)]
use std::intrinsics::{fmuladdf32, fmuladdf64};

fn main() {
    let mut saw_zero = false;
    let mut saw_nonzero = false;
    for _ in 0..50 {
        let a = std::hint::black_box(0.1_f64);
        let b = std::hint::black_box(0.2);
        let c = std::hint::black_box(-a * b);
        // It is unspecified whether the following operation is fused or not. The
        // following evaluates to 0.0 if unfused, and nonzero (-1.66e-18) if fused.
        let x = unsafe { fmuladdf64(a, b, c) };
        if x == 0.0 {
            saw_zero = true;
        } else {
            saw_nonzero = true;
        }
    }
    assert!(
        saw_zero && saw_nonzero,
        "`fmuladdf64` failed to be evaluated as both fused and unfused"
    );

    let mut saw_zero = false;
    let mut saw_nonzero = false;
    for _ in 0..50 {
        let a = std::hint::black_box(0.1_f32);
        let b = std::hint::black_box(0.2);
        let c = std::hint::black_box(-a * b);
        // It is unspecified whether the following operation is fused or not. The
        // following evaluates to 0.0 if unfused, and nonzero (-8.1956386e-10) if fused.
        let x = unsafe { fmuladdf32(a, b, c) };
        if x == 0.0 {
            saw_zero = true;
        } else {
            saw_nonzero = true;
        }
    }
    assert!(
        saw_zero && saw_nonzero,
        "`fmuladdf32` failed to be evaluated as both fused and unfused"
    );
}
