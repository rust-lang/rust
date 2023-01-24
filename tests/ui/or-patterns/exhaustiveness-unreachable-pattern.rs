#![deny(unreachable_patterns)]

// We wrap patterns in a tuple because top-level or-patterns were special-cased.
fn main() {
    match (0u8,) {
        (1 | 2,) => {}
        (1,) => {}
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
        _ => {}
    }
    match (0u8,) {
        (1 | 2,) => {}
        (2,) => {}
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
        _ => {}
    }
    match (0u8,) {
        (1,) => {}
        (2,) => {}
        (1 | 2,) => {}
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
        _ => {}
    }
    match (0u8, 0u8) {
        (1 | 2, 3 | 4) => {}
        (1, 3) => {}
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
        (1, 4) => {}
        //~^ this arm is never executed
        (2, 4) => {}
        //~^ this arm is never executed
        (2 | 1, 4) => {}
        //~^ this arm is never executed
        (1, 5 | 6) => {}
        (1, 4 | 5) => {}
        //~^ this arm is never executed
        _ => {}
    }
    match (true, true) {
        (false | true, false | true) => (),
    }
    match (Some(0u8),) {
        (None | Some(1 | 2),) => {}
        (Some(1),) => {}
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
        (None,) => {}
        //~^ this arm is never executed
        _ => {}
    }
    match ((0u8,),) {
        ((1 | 2,) | (3 | 4,),) => {}
        ((1..=4,),) => {}
        //~^ ERROR unreachable pattern
        //~| this arm is never executed
        _ => {}
    }

    match (0,) {
        (1 | 1,) => {}
        //~^ ERROR unreachable pattern
        //~| this pattern is unreachable
        _ => {}
    }
    match 0 {
        (0 | 1) | 1 => {}
        //~^ ERROR unreachable pattern
        //~| this pattern is unreachable
        _ => {}
    }
    match 0 {
        // We get two errors because recursive or-pattern expansion means we don't notice the two
        // errors span a whole pattern. This could be better but doesn't matter much
        0 | (0 | 0) => {}
        //~^ ERROR unreachable pattern
        //~| this pattern is unreachable
        _ => {}
    }
    match None {
        // There is only one error that correctly points to the whole subpattern
        Some(0) | Some(0 | 0) => {}
        //~^^ ERROR unreachable pattern
        //~| this pattern is unreachable
        _ => {}
    }
    match [0; 2] {
        [0
            | 0
        //~^ ERROR unreachable pattern
        //~| this pattern is unreachable
        , 0
            | 0] => {}
        //~^ this pattern is unreachable
        _ => {}
    }
    match &[][..] {
        [0] => {}
        [0, _] => {}
        [0, _, _] => {}
        [1, ..] => {}
        [1
        //~^ ERROR unreachable pattern
        //~| this pattern is unreachable
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
        [true
        //~^ ERROR unreachable pattern
        //~| this pattern is unreachable
            | false, ..] => {}
        _ => {}
    }
    match (true, None) {
        (true, Some(_)) => {}
        (false, Some(true)) => {}
        (true | false, None | Some(true
        //~^ ERROR unreachable pattern
        //~| this pattern is unreachable
                                   | false)) => {}
    }
    macro_rules! t_or_f {
        () => {
            (true
            //~^ ERROR unreachable pattern
            //~| this pattern is unreachable
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
        Some(0
        //~^ ERROR unreachable pattern
        //~| this pattern is unreachable
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
        None | Some(true | false) => {} //~ ERROR unreachable
    }

    // A subpattern that is unreachable in all branches is overall unreachable.
    match (true, true) {
        (false, true) => {}
        (true, true) => {}
        (false | true, false | true) => {} //~^ ERROR unreachable pattern
                                           //~| this pattern is unreachable
    }
    match (true, true) {
        (true, false) => {}
        (true, true) => {}
        (
            false | true,
            //~^ ERROR unreachable pattern
            //~| this pattern is unreachable
            false | true,
        ) => {}
    }
}
