// should-ice
#![allow(incomplete_features)]
#![feature(or_patterns)]
#![deny(unreachable_patterns)]

// The ice will get removed once or-patterns are correctly implemented
fn main() {
    // We wrap patterns in a tuple because top-level or-patterns are special-cased for now.
    match (0u8,) {
        (1 | 2,) => {}
        //~^ ERROR simplifyable pattern found
        // This above is the ICE error message
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
}
