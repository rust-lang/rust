// build-pass (FIXME(62277): could be check-pass?)

#![feature(exclusive_range_pattern)]
#![warn(unreachable_patterns)]

fn main() {
    // These cases should generate no warning.
    match 10 {
        1..10 => {},
        10 => {},
        _ => {},
    }

    match 10 {
        1..10 => {},
        9..=10 => {},
        _ => {},
    }

    match 10 {
        1..10 => {},
        10..=10 => {},
        _ => {},
    }

    // These cases should generate an "unreachable pattern" warning.
    match 10 {
        1..10 => {},
        9 => {},
        _ => {},
    }

    match 10 {
        1..10 => {},
        8..=9 => {},
        _ => {},
    }

    match 10 {
        1..10 => {},
        9..=9 => {},
        _ => {},
    }
}
