// normalize-stderr-32bit: "-2147483648isize" -> "$$ISIZE_MIN"
// normalize-stderr-64bit: "-9223372036854775808isize" -> "$$ISIZE_MIN"
// normalize-stderr-32bit: "2147483647isize" -> "$$ISIZE_MAX"
// normalize-stderr-64bit: "9223372036854775807isize" -> "$$ISIZE_MAX"
// normalize-stderr-32bit: "4294967295usize" -> "$$USIZE_MAX"
// normalize-stderr-64bit: "18446744073709551615usize" -> "$$USIZE_MAX"

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
