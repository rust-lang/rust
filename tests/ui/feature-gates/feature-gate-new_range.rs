#![feature(new_range_api)]

fn main() {
    let a: core::range::RangeFrom<u8> = 1..;
    //~^ ERROR mismatched types
    let b: core::range::Range<u8> = 2..3;
    //~^ ERROR mismatched types
    let c: core::range::RangeInclusive<u8> = 4..=5;
    //~^ ERROR mismatched types
}
