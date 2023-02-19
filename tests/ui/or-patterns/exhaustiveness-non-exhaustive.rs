#![deny(unreachable_patterns)]

// We wrap patterns in a tuple because top-level or-patterns were special-cased.
fn main() {
    match (0u8, 0u8) {
        //~^ ERROR match is non-exhaustive
        (0 | 1, 2 | 3) => {}
    }
    match ((0u8,),) {
        //~^ ERROR match is non-exhaustive
        ((0 | 1,) | (2 | 3,),) => {}
    }
    match (Some(0u8),) {
        //~^ ERROR match is non-exhaustive
        (None | Some(0 | 1),) => {}
    }
}
