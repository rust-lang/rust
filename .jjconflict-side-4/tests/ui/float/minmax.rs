//FIXME(llvm21) This should be a library test, but old LLVM miscompiles things so we can't just
// test this properly everywhere. Once we require LLVM 22, remove this test and enable the
// commented-out tests in `library/coretests/tests/floats/mod.rs` instead.
//@ min-llvm-version: 22
//@ run-pass

use std::hint::black_box;

const SNAN32: f32 = f32::from_bits(f32::NAN.to_bits() - 1);
const SNAN64: f64 = f64::from_bits(f64::NAN.to_bits() - 1);

fn main() {
    assert_eq!(SNAN32.min(black_box(9.0)), 9.0f32);
    assert_eq!(black_box(SNAN32).min(-9.0), -9.0f32);
    assert_eq!((9.0f32).min(black_box(SNAN32)), 9.0f32);
    assert_eq!(black_box(-9.0f32).min(SNAN32), -9.0f32);

    assert_eq!(SNAN64.min(black_box(9.0)), 9.0f64);
    assert_eq!(black_box(SNAN64).min(-9.0), -9.0f64);
    assert_eq!((9.0f64).min(black_box(SNAN64)), 9.0f64);
    assert_eq!(black_box(-9.0f64).min(SNAN64), -9.0f64);
}
