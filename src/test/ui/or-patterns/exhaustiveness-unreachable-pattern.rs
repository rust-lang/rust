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
    match (true, true) {
        (false | true, false | true) => (),
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

    // A subpattern that is only unreachable in one branch is overall reachable.
    match (true, true) {
        (true, true) => {}
        (false | true, false | true) => {}
    }
    match (true, true) {
        (true, true) => {}
        (false, false) => {}
        (false | true, false | true) => {}
    }
    // https://github.com/rust-lang/rust/issues/76836
    match None {
        Some(false) => {}
        None | Some(true
                | false) => {} //~ ERROR unreachable
    }

    // A subpattern that is unreachable in all branches is overall unreachable.
    match (true, true) {
        (false, true) => {}
        (true, true) => {}
        (false | true, false
            | true) => {} //~ ERROR unreachable
    }
    match (true, true) {
        (true, false) => {}
        (true, true) => {}
        (false
            | true, //~ ERROR unreachable
            false | true) => {}
    }
}
