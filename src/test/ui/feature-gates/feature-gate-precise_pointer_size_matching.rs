#![feature(exclusive_range_pattern)]

fn main() {
    match 0usize { //~ERROR non-exhaustive patterns: `_` not covered
        0..=usize::MAX => {}
    }

    match 0isize { //~ERROR non-exhaustive patterns: `_` not covered
        1..=20 => {}
        -5..3 => {}
    }
}
