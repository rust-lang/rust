// aux-build:lint_stability.rs

#![feature(unstable_test_feature_internal)]
//~^ ERROR the feature `unstable_test_feature_internal` is internal to the compiler or standard library
#![forbid(internal_features)]

extern crate lint_stability;

fn main() {
    let _ = lint_stability::InternalStruct;
}
