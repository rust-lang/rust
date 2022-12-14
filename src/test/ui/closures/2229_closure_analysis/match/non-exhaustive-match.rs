// edition:2021

// aux-build:match_non_exhaustive_lib.rs

/* The error message for non-exhaustive matches on non-local enums
 * marked as non-exhaustive should mention the fact that the enum
 * is marked as non-exhaustive (issue #85227).
 */

// Ignore non_exhaustive in the same crate
#[non_exhaustive]
enum L1 { A, B }
enum L2 { C }

extern crate match_non_exhaustive_lib;
use match_non_exhaustive_lib::{E1, E2, E3, E4};

fn foo() -> (L1, L2) {todo!()}
fn bar() -> (E1, E2, E3, E4) {todo!()}

fn main() {
    let (l1, l2) = foo();
    // No error for enums defined in this crate
    let _a = || { match l1 { L1::A => (), L1::B => () } };
    // (except if the match is already non-exhaustive)
    let _b = || { match l1 { L1::A => () } };
    //~^ ERROR: non-exhaustive patterns: `L1::B` not covered [E0004]

    // l2 should not be captured as it is a non-exhaustive SingleVariant
    // defined in this crate
    let _c = || { match l2 { L2::C => (), _ => () }  };
    let mut mut_l2 = l2;
    _c();

    // E1 is not visibly uninhabited from here
    let (e1, e2, e3, e4) = bar();
    let _d = || { match e1 {} };
    //~^ ERROR: non-exhaustive patterns: type `E1` is non-empty [E0004]
    let _e = || { match e2 { E2::A => (), E2::B => () } };
    //~^ ERROR: non-exhaustive patterns: `_` not covered [E0004]
    let _f = || { match e2 { E2::A => (), E2::B => (), _ => () }  };

    // e3 should be captured as it is a non-exhaustive SingleVariant
    // defined in another crate
    let _g = || { match e3 { E3::C => (), _ => () }  };
    let mut mut_e3 = e3;
    //~^ ERROR: cannot move out of `e3` because it is borrowed
    _g();

    // e4 should not be captured as it is a SingleVariant
    let _h = || { match e4 { E4::D => (), _ => () }  };
    let mut mut_e4 = e4;
    _h();
}
