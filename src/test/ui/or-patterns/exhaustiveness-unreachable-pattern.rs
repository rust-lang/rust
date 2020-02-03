#![feature(or_patterns)]
#![deny(unreachable_patterns)]

// We wrap patterns in a tuple because top-level or-patterns were special-cased.
fn main() {
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
    match (0u8, 0u8) {
        (1 | 2, 3 | 4) => {}
        (1, 3) => {}     //~ ERROR unreachable pattern
        (1, 4) => {}     //~ ERROR unreachable pattern
        (2, 4) => {}     //~ ERROR unreachable pattern
        (2 | 1, 4) => {} //~ ERROR unreachable pattern
        (1, 5 | 6) => {}
        (1, 4 | 5) => {} //~ ERROR unreachable pattern
        _ => {}
    }
    match (Some(0u8),) {
        (None | Some(1 | 2),) => {}
        (Some(1),) => {} //~ ERROR unreachable pattern
        (None,) => {}    //~ ERROR unreachable pattern
        _ => {}
    }
    match ((0u8,),) {
        ((1 | 2,) | (3 | 4,),) => {}
        ((1..=4,),) => {} //~ ERROR unreachable pattern
        _ => {}
    }

    match (0,) {
        (1 | 1,) => {} //~ ERROR unreachable
        _ => {}
    }
    match [0; 2] {
        [0
            | 0 //~ ERROR unreachable
        , 0
            | 0] => {} //~ ERROR unreachable
        _ => {}
    }
    match &[][..] {
        [0] => {}
        [0, _] => {}
        [0, _, _] => {}
        [1, ..] => {}
        [1 //~ ERROR unreachable
            | 2, ..] => {}
        _ => {}
    }
    match Some(0) {
        Some(0) => {}
        Some(0 //~ ERROR unreachable
             | 1) => {}
        _ => {}
    }
}
