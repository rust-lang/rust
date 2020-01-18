#![feature(or_patterns)]

#![allow(incomplete_features)]
#![deny(unreachable_patterns)]

// We wrap patterns in a tuple because top-level or-patterns are special-cased for now.
fn main() {
    // Get the fatal error out of the way
    match (0u8,) {
        (0 | _,) => {}
        //~^ ERROR or-patterns are not fully implemented yet
    }

    match (0u8, 0u8) {
        //~^ ERROR non-exhaustive patterns: `(2u8..=std::u8::MAX, _)`
        (0 | 1, 2 | 3) => {}
    }
    match ((0u8,),) {
        //~^ ERROR non-exhaustive patterns: `((4u8..=std::u8::MAX))`
        ((0 | 1,) | (2 | 3,),) => {},
    }
    match (Some(0u8),) {
        //~^ ERROR non-exhaustive patterns: `(Some(2u8..=std::u8::MAX))`
        (None | Some(0 | 1),) => {}
    }
}
