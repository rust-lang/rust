#![deny(unreachable_patterns)]

// We wrap patterns in a tuple because top-level or-patterns were special-cased.
#[rustfmt::skip]
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
    match (0u8, 0u8, 0u8) {
        (0, 0, 0) => {}
        (0, 0 | 1, 0) => {} //~ ERROR unreachable pattern
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
    match 0 {
        (0 | 1) | 1 => {} //~ ERROR unreachable
        _ => {}
    }
    match 0 {
        // We get two errors because recursive or-pattern expansion means we don't notice the two
        // errors span a whole pattern. This could be better but doesn't matter much
        0 | (0 | 0) => {}
        //~^ ERROR unreachable
        //~| ERROR unreachable
        _ => {}
    }
    match None {
        // There is only one error that correctly points to the whole subpattern
        Some(0) |
            Some( //~ ERROR unreachable
                0 | 0) => {}
        _ => {}
    }
    match [0; 2] {
        [0
            | 0 //~ ERROR unreachable
        , 0
            | 0] => {} //~ ERROR unreachable
        _ => {}
    }
    match (true, 0) {
        (true, 0 | 0) => {} //~ ERROR unreachable
        (_, 0 | 0) => {} //~ ERROR unreachable
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
    match &[][..] {
        [true] => {}
        [true | false, ..] => {}
        _ => {}
    }
    match &[][..] {
        [false] => {}
        [true, ..] => {}
        [true //~ ERROR unreachable
            | false, ..] => {}
        _ => {}
    }
    match (true, None) {
        (true, Some(_)) => {}
        (false, Some(true)) => {}
        (true | false, None | Some(true //~ ERROR unreachable
                                   | false)) => {}
    }
    macro_rules! t_or_f {
        () => {
            (true //~ ERROR unreachable
            | false)
        };
    }
    match (true, None) {
        (true, Some(_)) => {}
        (false, Some(true)) => {}
        (true | false, None | Some(t_or_f!())) => {}
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
    match (true, true) {
        (x, y)
            | (y, x) => {} //~ ERROR unreachable
    }
}

fn unreachable_in_param((_ | (_, _)): (bool, bool)) {}
//~^ ERROR unreachable

fn unreachable_in_binding() {
    let bool_pair = (true, true);
    let bool_option = Some(true);

    let (_ | (_, _)) = bool_pair;
    //~^ ERROR unreachable
    for (_ | (_, _)) in [bool_pair] {}
    //~^ ERROR unreachable

    let (Some(_) | Some(true)) = bool_option else { return };
    //~^ ERROR unreachable
    if let Some(_) | Some(true) = bool_option {}
    //~^ ERROR unreachable
    while let Some(_) | Some(true) = bool_option {}
    //~^ ERROR unreachable
}
