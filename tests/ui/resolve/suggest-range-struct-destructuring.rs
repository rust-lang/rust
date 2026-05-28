use std::ops::{Range, RangeFrom, RangeInclusive, RangeTo, RangeToInclusive};

fn test_range(r: Range<u32>) {
    let start..end = r;
    //~^ ERROR cannot find value `start`
    //~| ERROR cannot find value `end`
}

fn test_inclusive(r: RangeInclusive<u32>) {
    let start..=end = r;
    //~^ ERROR cannot find value `start`
    //~| ERROR cannot find value `end`
}

fn test_from(r: RangeFrom<u32>) {
    let start.. = r;
    //~^ ERROR cannot find value `start`
}

fn test_to(r: RangeTo<u32>) {
    let ..end = r;
    //~^ ERROR cannot find value `end`
}

fn test_to_inclusive(r: RangeToInclusive<u32>) {
    let ..=end = r;
    //~^ ERROR cannot find value `end`
}

// Case 6: Complex Path (Keep this! It works!)
mod my {
    // We don't define MISSING here to trigger the error
}
fn test_path(r: Range<u32>) {
    let my::MISSING..end = r;
    //~^ ERROR cannot find value `MISSING`
    //~| ERROR cannot find value `end`
}

fn main() {}
