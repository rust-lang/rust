//! Check that the flags to control the extra rounding error work.
//@revisions: random max none
//@[max]compile-flags: -Zmiri-max-extra-rounding-error
//@[none]compile-flags: -Zmiri-no-extra-rounding-error
#![feature(cfg_select)]

use std::collections::HashSet;
use std::hint::black_box;

fn main() {
    let expected = cfg_select! {
        random => 9, // -4 ..= +4 ULP error
        max => 2,
        none => 1,
    };
    // Call `sin(0.5)` a bunch of times and see how many different values we get.
    let mut values = HashSet::new();
    for _ in 0..(expected * 16) {
        let val = black_box(0.5f64).sin();
        values.insert(val.to_bits());
    }
    assert_eq!(values.len(), expected);

    if !cfg!(none) {
        // Ensure the smallest and biggest value are 8 ULP apart.
        // We can just subtract the raw bit representations for this.
        let min = *values.iter().min().unwrap();
        let max = *values.iter().max().unwrap();
        assert_eq!(min.abs_diff(max), 8);
    }
}
