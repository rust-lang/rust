//@ aux-build:hidden.rs

extern crate hidden;

use hidden::HiddenEnum;

enum InCrate {
    A,
    B,
    #[doc(hidden)]
    C,
}

fn main() {
    match HiddenEnum::A {
        HiddenEnum::A => {}
        HiddenEnum::B => {}
    }
    //~^^^^ ERROR non-exhaustive patterns: `_` not covered

    match HiddenEnum::A {
        HiddenEnum::A => {}
        HiddenEnum::C => {}
    }
    //~^^^^ ERROR non-exhaustive patterns: `HiddenEnum::B` not covered

    match HiddenEnum::A {
        HiddenEnum::A => {}
    }
    //~^^^ ERROR non-exhaustive patterns: `HiddenEnum::B` and `_` not covered

    match None {
        None => {}
        Some(HiddenEnum::A) => {}
    }
    //~^^^^ ERROR non-exhaustive patterns: `Some(HiddenEnum::B)` and `Some(_)` not covered

    match InCrate::A {
        InCrate::A => {}
        InCrate::B => {}
    }
    //~^^^^ ERROR non-exhaustive patterns: `InCrate::C` not covered
}
