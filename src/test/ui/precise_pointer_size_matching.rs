#![feature(precise_pointer_size_matching)]
#![feature(exclusive_range_pattern)]

#![deny(unreachable_patterns)]

use std::{usize, isize};

fn main() {
    match 0isize {
        isize::MIN ..= isize::MAX => {} // ok
    }

    match 0usize {
        0 ..= usize::MAX => {} // ok
    }

    match 0isize { //~ ERROR non-exhaustive patterns
        1 ..= 8 => {}
        -5 ..= 20 => {}
    }

    match 0usize { //~ ERROR non-exhaustive patterns
        1 ..= 8 => {}
        5 ..= 20 => {}
    }
}
