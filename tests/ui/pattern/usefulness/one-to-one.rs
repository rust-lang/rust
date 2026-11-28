//@ aux-build:non-exhaustive.rs

//! There is a fast path for matching a fieldless enum exactly as it is declared.
//! This tests that it correctly rejects some “near misses”.

extern crate non_exhaustive;
use non_exhaustive::NonExhaustiveEnum;

enum SimpleEnum {
    A,
    B,
    C,
}

pub fn main() {
    // Guard
    match SimpleEnum::A { //~ ERROR non-exhaustive patterns: `SimpleEnum::B` not covered
        SimpleEnum::A => {}
        SimpleEnum::B if false => {}
        SimpleEnum::C => {}
    }

    // Missing the final variant
    match SimpleEnum::A { //~ ERROR non-exhaustive patterns: `SimpleEnum::C` not covered
        SimpleEnum::A => {}
        SimpleEnum::B => {}
    }

    // Right number of arms, wrong variants
    match SimpleEnum::A { //~ ERROR non-exhaustive patterns: `SimpleEnum::A` not covered
        SimpleEnum::B => {}
        SimpleEnum::B => {}
        SimpleEnum::C => {}
    }

    // Right number of arms but not exhaustive
    match NonExhaustiveEnum::A { //~ ERROR non-exhaustive patterns: `_` not covered
        NonExhaustiveEnum::A => {}
        NonExhaustiveEnum::B => {}
    }
}
