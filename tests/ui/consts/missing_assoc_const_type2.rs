//! Test that we compute the right type for associated constants
//! of impls, even if the type is missing. We know it from the trait
//! declaration after all.

trait Range {
    const FIRST: u8;
    const LAST: u8;
}

struct TwoDigits;
impl Range for TwoDigits {
    const FIRST: _ = 10;
    //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for associated constants [E0121]
    const LAST: u8 = 99;
}

const FOOMP: [(); {
    TwoDigits::FIRST as usize
}] = [(); 10];

fn main() {}
