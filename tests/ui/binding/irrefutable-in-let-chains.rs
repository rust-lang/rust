// https://github.com/rust-lang/rust/issues/139369
// Test that the lint `irrefutable_let_patterns` now
// only checks single let binding.
//@ edition: 2024
//@ check-pass

#![feature(if_let_guard)]

use std::ops::Range;

fn main() {
    let opt = Some(None..Some(1));

    // test `if let`
    if let first = &opt {}
    //~^ WARN irrefutable `if let` pattern

    if let first = &opt && let Some(second) = first {}

    if let first = &opt && let (a, b) = (1, 2) {}

    if let first = &opt && let None = Some(1) {}

    if 4 * 2 == 0 && let first = &opt {}

    if let first = &opt
        && let Some(second) = first
        && let None = second.start
        && let v = 0
    {}

    if let Range { start: local_start, end: _ } = (None..Some(1)) {}
    //~^ WARN irrefutable `if let` pattern

    if let Range { start: local_start, end: _ } = (None..Some(1))
        && let None = local_start
    {}

    if let (a, b, c) = (Some(1), Some(1), Some(1)) {}
    //~^ WARN irrefutable `if let` pattern

    if let (a, b, c) = (Some(1), Some(1), Some(1)) && let None = Some(1) {}

    if let Some(ref first) = opt
        && let Range { start: local_start, end: _ } = first
        && let None = local_start
    {}

    // test `else if let`
    if opt == Some(None..None) {
    } else if let x = opt.clone().map(|_| 1) {
        //~^ WARN irrefutable `if let` pattern
    }

    if opt == Some(None..None) {
    } else if let x = opt.clone().map(|_| 1)
        && x == Some(1)
    {}

    if opt == Some(None..None) {
    } else if opt.is_some() && let x = &opt
    {}

    if opt == Some(None..None) {
    } else {
        if let x = opt.clone().map(|_| 1) && x == Some(1)
        {}
    }

    // test `if let guard`
    match opt {
        Some(ref first) if let second = first => {}
        //~^ WARN irrefutable `if let` guard pattern
        _ => {}
    }

    match opt {
        Some(ref first)
            if let second = first
                && let _third = second
                && let v = 4 + 4 => {}
        _ => {}
    }

    match opt {
        Some(ref first)
            if let Range { start: local_start, end: _ } = first
                && let None = local_start => {}
        _ => {}
    }

    match opt {
        Some(ref first)
            if let Range { start: Some(_), end: local_end } = first
                && let v = local_end
                && let w = v => {}
        _ => {}
    }

    // test `while let`
    while let first = &opt {}
    //~^ WARN irrefutable `while let` pattern

    while let first = &opt
        && let (a, b) = (1, 2)
    {}

    while let first = &opt
        && let Some(second) = first
        && let None = second.start
    {}

    while let Some(ref first) = opt
        && let second = first
        && let _third = second
    {}

    while let Some(ref first) = opt
        && let Range { start: local_start, end: _ } = first
        && let None = local_start
    {}
}
