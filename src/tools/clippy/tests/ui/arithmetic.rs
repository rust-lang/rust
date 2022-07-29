// run-rustfix

#![allow(clippy::unnecessary_owned_empty_strings)]
#![feature(saturating_int_impl)]
#![warn(clippy::arithmetic)]

use core::num::{Saturating, Wrapping};

pub fn hard_coded_allowed() {
    let _ = Saturating(0u32) + Saturating(0u32);
    let _ = String::new() + "";
    let _ = Wrapping(0u32) + Wrapping(0u32);

    let saturating: Saturating<u32> = Saturating(0u32);
    let string: String = String::new();
    let wrapping: Wrapping<u32> = Wrapping(0u32);

    let inferred_saturating = saturating + saturating;
    let inferred_string = string + "";
    let inferred_wrapping = wrapping + wrapping;

    let _ = inferred_saturating + inferred_saturating;
    let _ = inferred_string + "";
    let _ = inferred_wrapping + inferred_wrapping;
}

fn main() {}
