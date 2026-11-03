//! Ensure that on nightly we don't warn about using unstable target features.

//@ only-x86_64
// FIXME: Once GCC backend is fixed, remove this `ignore-backends`.
//@ ignore-backends: gcc

//@ compile-flags: -Ctarget-feature=+x87 --crate-type=rlib
//@ check-pass
//@ revisions: stable nightly
//@[stable] act-as-stable
//@[nightly] only-nightly

//[stable]~? WARNING unstable feature specified for `-Ctarget-feature`: `x87`
