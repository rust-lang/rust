//! Regression test for <https://github.com/rust-lang/rust/issues/164972>.
//!
//! At opt-level 0, AArch64 uses GlobalISel, which lowered saturating f16 to 128-bit integer
//! casts incorrectly: `-1.0f16 as i128` gave `18446744073709551615`.

//@ run-pass
//@ compile-flags: -Copt-level=0
// The fix is in LLVM 23.1.6.
//@ min-llvm-version: 23.1.6

#![feature(cfg_target_has_reliable_f16_f128)]
#![cfg_attr(target_has_reliable_f16, feature(f16))]
// expect the unexpected (`target_has_reliable_*` are not "known" configs since they are unstable)
#![expect(unexpected_cfgs)]

#[cfg(target_has_reliable_f16)]
fn check_f16() {
    use std::hint::black_box;

    assert_eq!(black_box(-1.0f16) as i128, -1);
    assert_eq!(black_box(f16::MIN) as i128, -65504);
    assert_eq!(black_box(f16::MAX) as i128, 65504);
    assert_eq!(black_box(f16::INFINITY) as i128, i128::MAX);
    assert_eq!(black_box(f16::NEG_INFINITY) as i128, i128::MIN);
    assert_eq!(black_box(f16::NAN) as i128, 0);

    assert_eq!(black_box(-1.0f16) as u128, 0);
    assert_eq!(black_box(f16::MAX) as u128, 65504);
    assert_eq!(black_box(f16::INFINITY) as u128, u128::MAX);
    assert_eq!(black_box(f16::NAN) as u128, 0);
}

fn main() {
    #[cfg(target_has_reliable_f16)]
    check_f16();
}
