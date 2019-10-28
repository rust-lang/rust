// check-pass
#![feature(exclusive_range_pattern)]
#![warn(unreachable_patterns)]
#![warn(overlapping_patterns)]

fn main() {
    // These cases should generate no warning.
    match 10 {
        1..10 => {},
        10 => {},
        _ => {},
    }

    match 10 {
        1..10 => {},
        9..=10 => {}, //~ WARNING multiple patterns covering the same range
        _ => {},
    }

    match 10 {
        1..10 => {},
        10..=10 => {},
        _ => {},
    }

    // These cases should generate "unreachable pattern" warnings.
    match 10 {
        1..10 => {},
        9 => {}, //~ WARNING unreachable pattern
        _ => {},
    }

    match 10 {
        1..10 => {},
        8..=9 => {}, //~ WARNING multiple patterns covering the same range
        _ => {},
    }

    match 10 {
        5..7 => {},
        6 => {}, //~ WARNING unreachable pattern
        1..10 => {}, //~ WARNING multiple patterns covering the same range
        9..=9 => {}, //~ WARNING unreachable pattern
        6 => {}, //~ WARNING unreachable pattern
        _ => {},
    }
}
