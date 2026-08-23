//! Test the "useless pattern" lint on or-pattern alternatives that are reachable but don't
//! influence the result of the match, e.g. `0` in `0 | _` (issue #160772).

#![deny(unreachable_patterns)]
#![allow(overlapping_range_endpoints)]

fn main() {
    match 0u8 {
        0 | _ => {}
        //~^ ERROR useless pattern
    }

    match 0u8 {
        0 | 0..=255 => {}
        //~^ ERROR useless pattern
    }

    // Covered by the union of the siblings, though neither covers it alone.
    match 0u8 {
        0..=1 | 1..=2 | 2..=3 => {}
        //~^ ERROR useless pattern
        _ => {}
    }

    match 0u8 {
        0 => {}
        0..=1 | 1 => {}
        //~^ ERROR useless pattern
        //~| ERROR unreachable pattern
        _ => {}
    }

    match 0u8 {
        0 | _ if false => {}
        //~^ ERROR useless pattern
        _ => {}
    }

    match 0u8 {
        0 if false => {}
        0 | _ => {}
        //~^ ERROR useless pattern
    }

    match Some(0u8) {
        Some(0 | _) => {}
        //~^ ERROR useless pattern
        None => {}
    }

    // No lint: which alternative matches determines `x`.
    match (0u8, 0u8) {
        (0, x) | (x, _) => {
            let _ = x;
        }
    }

    // Bindings outside the or-pattern bind the same value either way: still linted.
    match 0u8 {
        x @ (0 | _) => {
            //~^ ERROR useless pattern
            let _ = x;
        }
    }

    match 0u8 {
        0 | 0 if false => {}
        //~^ ERROR useless pattern
        //~| ERROR useless pattern
        _ => {}
    }

    // No lint: each alternative matches values the other doesn't.
    match 0u8 {
        0..=1 | 1..=2 => {}
        _ => {}
    }
}
