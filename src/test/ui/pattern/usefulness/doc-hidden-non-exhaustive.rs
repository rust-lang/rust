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
    //~^^^^ non-exhaustive patterns: `_` not covered

    match HiddenEnum::A {
        HiddenEnum::A => {}
        HiddenEnum::C => {}
    }
    //~^^^^ non-exhaustive patterns: `B` not covered

    match HiddenEnum::A {
        HiddenEnum::A => {}
    }
    //~^^^ non-exhaustive patterns: `B` and `_` not covered

    match None {
        None => {}
        Some(HiddenEnum::A) => {}
    }
    //~^^^^ non-exhaustive patterns: `Some(B)` and `Some(_)` not covered

    match InCrate::A {
        InCrate::A => {}
        InCrate::B => {}
    }
    //~^^^^ non-exhaustive patterns: `C` not covered
}
