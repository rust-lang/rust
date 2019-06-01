#![feature(exclusive_range_pattern)]
#![warn(clippy::match_overlapping_arm)]
#![allow(clippy::redundant_pattern_matching)]

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

    if let None = Some(42) {
        // nothing
    } else if let None = Some(42) {
        // another nothing :-)
    }
}

fn main() {}
