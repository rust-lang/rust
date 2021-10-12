// Test that the `non_exhaustive_omitted_patterns` lint is triggered correctly with variants
// marked stable and unstable.

#![feature(non_exhaustive_omitted_patterns_lint)]

// aux-build:unstable.rs
extern crate unstable;

use unstable::{UnstableEnum, OnlyUnstableEnum};

fn main() {
    // OK: this matches all the stable variants
    match UnstableEnum::Stable {
        UnstableEnum::Stable => {}
        UnstableEnum::Stable2 => {}
        #[deny(non_exhaustive_omitted_patterns)]
        _ => {}
    }

    match UnstableEnum::Stable {
        UnstableEnum::Stable => {}
        #[deny(non_exhaustive_omitted_patterns)]
        _ => {}
    }
    //~^^ some variants are not matched explicitly

    // Ok: although this is a bit odd, we don't have anything to report
    // since there is no stable variants and the feature is off
    #[deny(non_exhaustive_omitted_patterns)]
    match OnlyUnstableEnum::new() {
        _ => {}
    }
}
