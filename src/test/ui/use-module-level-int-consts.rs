// run-pass

// Make sure the module level constants are still there and accessible even after
// the corresponding associated constants have been added, and later stabilized.
#![allow(deprecated, deprecated_in_future)]
use std::{u16, f32};

fn main() {
    let _ = u16::MAX;
    let _ = f32::EPSILON;
    let _ = std::f64::MANTISSA_DIGITS;
}
