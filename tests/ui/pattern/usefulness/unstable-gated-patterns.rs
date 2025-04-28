#![feature(unstable_test_feature)]

//@ aux-build:unstable.rs

extern crate unstable;

use unstable::UnstableEnum;

fn main() {
    match UnstableEnum::Stable {
        UnstableEnum::Stable => {}
        UnstableEnum::Stable2 => {}
    }
    //~^^^^ ERROR non-exhaustive patterns: `UnstableEnum::Unstable` not covered

    // Ok: all variants are explicitly matched
    match UnstableEnum::Stable {
        UnstableEnum::Stable => {}
        UnstableEnum::Stable2 => {}
        UnstableEnum::Unstable => {}
    }
}
