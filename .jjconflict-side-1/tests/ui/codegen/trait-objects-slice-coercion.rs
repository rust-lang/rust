//! Regression test for <https://github.com/rust-lang/rust/issues/41744>.
//! This used to trigger LLVM assertion
//! `SrcTy must be larger than DestTy for Trunc`
//! because of a redundant truncate call when value is boolean.
//@ run-pass

trait Tc {}
impl Tc for bool {}

fn main() {
    let _: &[&dyn Tc] = &[&true];
}
