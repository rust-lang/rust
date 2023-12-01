// aux-build:normal-const-fn.rs
// check-pass
#![crate_type = "lib"]
#![feature(staged_api)]
#![feature(rustc_attrs)]
#![feature(rustc_private)]
#![allow(internal_features)]
#![feature(rustc_allow_const_fn_unstable)]
#![stable(feature = "stable_feature", since = "1.0.0")]

extern crate normal_const_fn;

// This ensures std can call const functions in it's deps that don't have
// access to rustc_const_stable annotations (and hense don't have a feature)
// gate.

#[rustc_const_stable(feature = "stable_feature", since = "1.0.0")]
#[stable(feature = "stable_feature", since = "1.0.0")]
#[rustc_allow_const_fn_unstable(any_unmarked)]
pub const fn do_something() {
    normal_const_fn::do_something_else()
}
