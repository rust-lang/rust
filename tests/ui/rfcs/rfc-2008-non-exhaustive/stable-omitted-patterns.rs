// Test that the `non_exhaustive_omitted_patterns` lint is triggered correctly with variants
// marked stable and unstable.

#![feature(non_exhaustive_omitted_patterns_lint)]

//@ aux-build:unstable.rs
extern crate unstable;

use unstable::{OnlyUnstableEnum, OnlyUnstableStruct, UnstableEnum, UnstableStruct};

fn main() {
    // OK: this matches all the stable variants
    #[deny(non_exhaustive_omitted_patterns)]
    match UnstableEnum::Stable {
        UnstableEnum::Stable => {}
        UnstableEnum::Stable2 => {}
        _ => {}
    }

    #[deny(non_exhaustive_omitted_patterns)]
    match UnstableEnum::Stable {
        //~^ ERROR some variants are not matched explicitly
        UnstableEnum::Stable => {}
        _ => {}
    }

    // Ok: although this is a bit odd, we don't have anything to report
    // since there is no stable variants and the feature is off
    #[deny(non_exhaustive_omitted_patterns)]
    match OnlyUnstableEnum::new() {
        _ => {}
    }

    // Ok: Same as the above enum (no fields can be matched on)
    #[warn(non_exhaustive_omitted_patterns)]
    let OnlyUnstableStruct { .. } = OnlyUnstableStruct::new();

    #[warn(non_exhaustive_omitted_patterns)]
    let UnstableStruct { stable, .. } = UnstableStruct::default();
    //~^ WARN some fields are not explicitly listed

    // OK: stable field is matched
    #[warn(non_exhaustive_omitted_patterns)]
    let UnstableStruct { stable, stable2, .. } = UnstableStruct::default();
}
