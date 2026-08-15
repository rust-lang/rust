//@ aux-build:match_non_exhaustive_lib.rs

/* The error message for non-exhaustive matches on non-local enums
 * marked as non-exhaustive should mention the fact that the enum
 * is marked as non-exhaustive (issue #85227).
 */

// Ignore non_exhaustive in the same crate
#[non_exhaustive]
enum L { A, B }

extern crate match_non_exhaustive_lib;
use match_non_exhaustive_lib::{E1, E2};

fn foo() -> L {todo!()}
fn bar() -> (E1, E2) {todo!()}

fn main() {
    let l = foo();
    // No error for enums defined in this crate
    match l { L::A => (), L::B => () };
    // (except if the match is already non-exhaustive)
    match l { L::A => () };
    //~^ ERROR: non-exhaustive patterns: `L::B` not covered [E0004]

    // E1 is not visibly uninhabited from here
    let (e1, e2) = bar();
    match e1 {};
    //~^ ERROR: non-exhaustive patterns: type `E1` is non-empty [E0004]
    match e2 { E2::A => (), E2::B => () };
    //~^ ERROR: non-exhaustive patterns: `_` not covered [E0004]
}
