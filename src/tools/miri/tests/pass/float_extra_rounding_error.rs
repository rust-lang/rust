//! Check that the flags to control the extra rounding error work.
//@revisions: random max none
//@[max]compile-flags: -Zmiri-max-extra-rounding-error
//@[none]compile-flags: -Zmiri-no-extra-rounding-error
#![feature(cfg_select)]

use std::collections::HashSet;
use std::hint::black_box;

fn main() {
    let expected = cfg_select! {
        random => 13, // FIXME: why is it 13?
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
}
