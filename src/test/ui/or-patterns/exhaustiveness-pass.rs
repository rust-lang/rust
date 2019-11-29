#![feature(or_patterns)]
#![feature(slice_patterns)]
#![allow(incomplete_features)]
#![deny(unreachable_patterns)]

// We wrap patterns in a tuple because top-level or-patterns are special-cased for now.
fn main() {
    // Get the fatal error out of the way
    match (0,) {
        (0 | _,) => {}
        //~^ ERROR or-patterns are not fully implemented yet
    }

    match (0,) {
        (1 | 2,) => {}
        _ => {}
    }

    match (0, 0) {
        (1 | 2, 3 | 4) => {}
        (1, 2) => {}
        (3, 1) => {}
        _ => {}
    }
    match (Some(0u8),) {
        (None | Some(0 | 1),) => {}
        (Some(2..=255),) => {}
    }
    match ((0,),) {
        ((0 | 1,) | (2 | 3,),) => {},
        ((_,),) => {},
    }
    match (&[0u8][..],) {
        ([] | [0 | 1..=255] | [_, ..],) => {},
    }

    match ((0, 0),) {
        ((0, 0) | (0, 1),) => {}
        _ => {}
    }
    match ((0, 0),) {
        ((0, 0) | (1, 0),) => {}
        _ => {}
    }

     // FIXME(or_patterns): Redundancies not detected for now.
    match (0,) {
        (1 | 1,) => {}
        _ => {}
    }
    match [0; 2] {
        [0 | 0, 0 | 0] => {}
        _ => {}
    }
    match &[][..] {
        [0] => {}
        [0, _] => {}
        [0, _, _] => {}
        [1, ..] => {}
        [1 | 2, ..] => {}
        _ => {}
    }
    match Some(0) {
        Some(0) => {}
        Some(0 | 1) => {}
        _ => {}
    }
}
