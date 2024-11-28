// Prefix in imports with empty braces should be resolved and checked privacy, stability, etc.

//@ aux-build:lint-stability.rs

extern crate lint_stability;

use lint_stability::UnstableEnum::{};
//~^ ERROR use of unstable library feature `unstable_test_feature`
use lint_stability::StableEnum::{}; // OK

fn main() {}
