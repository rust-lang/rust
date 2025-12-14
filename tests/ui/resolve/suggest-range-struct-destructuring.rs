use std::ops::Range;

fn test_basic_range(r: Range<u32>) {
    let start..end = r;
    //~^ ERROR cannot find value `start` in this scope
    //~| ERROR cannot find value `end` in this scope
}

fn test_different_names(r: Range<u32>) {
    let min..max = r;
    //~^ ERROR cannot find value `min` in this scope
    //~| ERROR cannot find value `max` in this scope
}

fn main() {}
