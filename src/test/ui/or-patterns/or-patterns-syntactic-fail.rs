// Test some cases where or-patterns may ostensibly be allowed but are in fact not.
// This is not a semantic test. We only test parsing.

#![feature(or_patterns)]

fn main() {}

enum E { A, B }
use E::*;

fn no_top_level_or_patterns() {
    // We do *not* allow or-patterns at the top level of lambdas...
    let _ = |A | B: E| (); //~ ERROR no implementation for `E | ()`
    //           -------- This looks like an or-pattern but is in fact `|A| (B: E | ())`.

    // ...and for now neither do we allow or-patterns at the top level of functions.
    fn fun1(A | B: E) {} //~ ERROR an or-pattern parameter must be wrapped in parentheses

    fn fun2(| A | B: E) {}
    //~^ ERROR an or-pattern parameter must be wrapped in parentheses
}
