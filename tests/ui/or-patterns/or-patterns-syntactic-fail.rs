// Test some cases where or-patterns may ostensibly be allowed but are in fact not.
// This is not a semantic test. We only test parsing.

fn main() {}

enum E { A, B }
use E::*;

fn no_top_level_or_patterns() {
    // We do *not* allow or-patterns at the top level of lambdas...
    let _ = |A | B: E| ();
                           //~^ ERROR expected identifier, found
    //           -------- This looks like an or-pattern but is in fact `|A| (B: E | ())`.
}

fn no_top_level_or_patterns_2() {
    // ...and for now neither do we allow or-patterns at the top level of functions.
    fn fun1(A | B: E) {}
    //~^ ERROR function parameters require top-level or-patterns in parentheses

    fn fun2(| A | B: E) {}
    //~^ ERROR function parameters require top-level or-patterns in parentheses

    // We don't allow top-level or-patterns before type annotation in let-statements because we
    // want to reserve this syntactic space for possible future type ascription.
    let A | B: E = A;
    //~^ ERROR `let` bindings require top-level or-patterns in parentheses

    let | A | B: E = A;
    //~^ ERROR `let` bindings require top-level or-patterns in parentheses

    let (A | B): E = A; // ok -- wrapped in parens
}
