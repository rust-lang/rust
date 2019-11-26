#![feature(or_patterns)]
#![feature(slice_patterns)]
#![allow(incomplete_features)]
#![deny(unreachable_patterns)]

fn main() {
    // We wrap patterns in a tuple because top-level or-patterns are special-cased for now.

    // Get the fatal error out of the way
    match (0u8,) {
        (0 | _,) => {}
        //~^ ERROR or-patterns are not fully implemented yet
    }

    match (0u8,) {
        (1 | 2,) => {}
        _ => {}
    }

    match (0u8,) {
        (1 | 2,) => {}
        (1,) => {} //~ ERROR unreachable pattern
        _ => {}
    }
    match (0u8,) {
        (1 | 2,) => {}
        (2,) => {} //~ ERROR unreachable pattern
        _ => {}
    }
    match (0u8,) {
        (1,) => {}
        (2,) => {}
        (1 | 2,) => {} //~ ERROR unreachable pattern
        _ => {}
    }
    match (0u8,) {
        (1 | 1,) => {} // redundancy not detected for now
        _ => {}
    }
    match (0u8, 0u8) {
        (1 | 2, 3 | 4) => {}
        (1, 2) => {}
        (1, 3) => {} //~ ERROR unreachable pattern
        (1, 4) => {} //~ ERROR unreachable pattern
        (2, 4) => {} //~ ERROR unreachable pattern
        (2 | 1, 4) => {} //~ ERROR unreachable pattern
        _ => {}
    }
    match (Some(0u8),) {
        (None | Some(1 | 2),) => {}
        (Some(1),) => {} //~ ERROR unreachable pattern
        (None,) => {} //~ ERROR unreachable pattern
        (Some(_),) => {}
    }
    match ((0u8,),) {
        ((1 | 2,) | (3 | 4,),) => {},
        ((1..=4,),) => {}, //~ ERROR unreachable pattern
        ((_,),) => {},
    }
    match (&[0u8][..],) {
        ([] | [0 | 1..=255] | [_, ..],) => {},
        (_,) => {}, //~ ERROR unreachable pattern
    }
}
