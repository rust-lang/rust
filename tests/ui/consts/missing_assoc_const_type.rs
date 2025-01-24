//! Test that we compute the right type for associated constants
//! of impls, even if the type is missing. We know it from the trait
//! declaration after all.

trait Range {
    const FIRST: u8;
    const LAST: u8;
}

struct TwoDigits;
impl Range for TwoDigits {
    const FIRST:  = 10; //~ ERROR missing type for `const` item
    const LAST: u8 = 99;
}

const fn digits(x: u8) -> usize {
    match x {
        TwoDigits::FIRST..=TwoDigits::LAST => 0, // ok, `const` error already emitted
        0..=9 | 100..=255 => panic!(),
    }
}

const FOOMP: [(); {
    digits(42)
}] = [];

fn main() {}
