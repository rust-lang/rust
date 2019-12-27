#![feature(or_patterns)]
#![deny(unreachable_patterns)]

// check-pass

// We wrap patterns in a tuple because top-level or-patterns were special-cased.
fn main() {
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
        ((0 | 1,) | (2 | 3,),) => {}
        ((_,),) => {}
    }
    match (&[0u8][..],) {
        ([] | [0 | 1..=255] | [_, ..],) => {}
    }

    match ((0, 0),) {
        ((0, 0) | (0, 1),) => {}
        _ => {}
    }
    match ((0, 0),) {
        ((0, 0) | (1, 0),) => {}
        _ => {}
    }
}
