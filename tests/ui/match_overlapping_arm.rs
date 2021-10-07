#![feature(exclusive_range_pattern)]
#![feature(half_open_range_patterns)]
#![warn(clippy::match_overlapping_arm)]
#![allow(clippy::redundant_pattern_matching)]
#![allow(clippy::if_same_then_else, clippy::equatable_if_let)]

/// Tests for match_overlapping_arm

fn overlapping() {
    const FOO: u64 = 2;

    match 42 {
        0..=10 => println!("0 ... 10"),
        0..=11 => println!("0 ... 11"),
        _ => (),
    }

    match 42 {
        0..=5 => println!("0 ... 5"),
        6..=7 => println!("6 ... 7"),
        FOO..=11 => println!("0 ... 11"),
        _ => (),
    }

    match 42 {
        2 => println!("2"),
        0..=5 => println!("0 ... 5"),
        _ => (),
    }

    match 42 {
        2 => println!("2"),
        0..=2 => println!("0 ... 2"),
        _ => (),
    }

    match 42 {
        0..=10 => println!("0 ... 10"),
        11..=50 => println!("11 ... 50"),
        _ => (),
    }

    match 42 {
        2 => println!("2"),
        0..2 => println!("0 .. 2"),
        _ => (),
    }

    match 42 {
        0..10 => println!("0 .. 10"),
        10..50 => println!("10 .. 50"),
        _ => (),
    }

    match 42 {
        0..11 => println!("0 .. 11"),
        0..=11 => println!("0 ... 11"),
        _ => (),
    }

    match 42 {
        5..7 => println!("5 .. 7"),
        0..10 => println!("0 .. 10"),
        _ => (),
    }

    match 42 {
        5..10 => println!("5 .. 10"),
        0..=10 => println!("0 ... 10"),
        _ => (),
    }

    match 42 {
        0..14 => println!("0 .. 14"),
        5..10 => println!("5 .. 10"),
        _ => (),
    }

    match 42 {
        5..14 => println!("5 .. 14"),
        0..=10 => println!("0 ... 10"),
        _ => (),
    }

    match 42 {
        0..7 => println!("0 .. 7"),
        0..=10 => println!("0 ... 10"),
        _ => (),
    }

    /*
    // FIXME(JohnTitor): uncomment this once rustfmt knows half-open patterns
    match 42 {
        0.. => println!("0 .. 42"),
        3.. => println!("3 .. 42"),
        _ => (),
    }

    match 42 {
        ..=23 => println!("0 ... 23"),
        ..26 => println!("0 .. 26"),
        _ => (),
    }
    */

    if let None = Some(42) {
        // nothing
    } else if let None = Some(42) {
        // another nothing :-)
    }
}

fn main() {}
