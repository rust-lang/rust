//! Make sure the module level constants are still there and accessible even after
//! the corresponding associated constants have been added, and later stabilized.

//@ run-pass

#![allow(deprecated, deprecated_in_future)]
use std::{f32, u16};

fn main() {
    let _ = u16::MAX;
    let _ = f32::EPSILON;
    let _ = std::f64::MANTISSA_DIGITS;
}
