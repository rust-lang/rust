// aux-build:hidden.rs

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
    //~^^^^ match is non-exhaustive

    match HiddenEnum::A {
        HiddenEnum::A => {}
        HiddenEnum::C => {}
    }
    //~^^^^ match is non-exhaustive

    match HiddenEnum::A {
        HiddenEnum::A => {}
    }
    //~^^^ match is non-exhaustive

    match None {
        None => {}
        Some(HiddenEnum::A) => {}
    }
    //~^^^^ match is non-exhaustive

    match InCrate::A {
        InCrate::A => {}
        InCrate::B => {}
    }
    //~^^^^ match is non-exhaustive
}
