#![feature(or_patterns)]
#![feature(slice_patterns)]
#![allow(incomplete_features)]
#![deny(unreachable_patterns)]

// We wrap patterns in a tuple because top-level or-patterns are special-cased for now.
fn main() {
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
        (1 | 1,) => {} // FIXME(or_patterns): redundancy not detected for now.
        _ => {}
    }
    match (0u8, 0u8) {
        (1 | 2, 3 | 4) => {}
        (1, 2) => {}
        (2, 1) => {}
        _ => {}
    }
    match (Some(0u8),) {
        (None | Some(0 | 1),) => {}
        (Some(2..=255),) => {}
    }
    match ((0u8,),) {
        ((0 | 1,) | (2 | 3,),) => {},
        ((_,),) => {},
    }
    match (&[0u8][..],) {
        ([] | [0 | 1..=255] | [_, ..],) => {},
    }
}
