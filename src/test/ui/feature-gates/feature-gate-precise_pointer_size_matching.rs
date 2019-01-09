#![feature(exclusive_range_pattern)]

use std::usize::MAX;

fn main() {
    match 0usize { //~ERROR non-exhaustive patterns: `_` not covered
        0..=MAX => {}
    }

    match 0isize { //~ERROR non-exhaustive patterns: `_` not covered
        1..=20 => {}
        -5..3 => {}
    }
}
