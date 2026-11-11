#![feature(pattern_types, structural_match)]

use std::pat::pattern_type;

const THREE: pattern_type!(u32 is 1..) = 3;

const _: () = match THREE {
    THREE => {}
    //~^ ERROR non-structural type
    _ => unreachable!(),
};

const _: () = match THREE {
    3 => {}
    //~^ ERROR mismatched types
    _ => unreachable!(),
};

const _: () = match 3 {
    THREE => {}
    //~^ ERROR mismatched types
    _ => unreachable!(),
};

fn main() {}
