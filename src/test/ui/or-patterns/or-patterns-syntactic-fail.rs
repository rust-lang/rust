// Test some cases where or-patterns may ostensibly be allowed but are in fact not.
// This is not a semantic test. We only test parsing.

#![feature(or_patterns)]
//~^ WARN the feature `or_patterns` is incomplete and may cause the compiler to crash

fn main() {}

// Test the `pat` macro fragment parser:
macro_rules! accept_pat {
    ($p:pat) => {}
}

accept_pat!(p | q); //~ ERROR no rules expected the token `|`
accept_pat!(| p | q); //~ ERROR no rules expected the token `|`

// Non-macro tests:

enum E { A, B }
use E::*;

fn no_top_level_or_patterns() {
    // We do *not* allow or-patterns at the top level of lambdas...
    let _ = |A | B: E| (); //~ ERROR binary operation `|` cannot be applied to type `E`
    //           -------- This looks like an or-pattern but is in fact `|A| (B: E | ())`.

    // ...and for now neither do we allow or-patterns at the top level of functions.
    fn fun(A | B: E) {} //~ ERROR expected one of `:` or `@`, found `|`
}

// We also do not allow a leading `|` when not in a top level position:

#[cfg(FALSE)]
fn no_leading_parens() {
    let ( | A | B); //~ ERROR expected pattern, found `|`
}

#[cfg(FALSE)]
fn no_leading_tuple() {
    let ( | A | B,); //~ ERROR expected pattern, found `|`
}

#[cfg(FALSE)]
fn no_leading_slice() {
    let [ | A | B ]; //~ ERROR expected pattern, found `|`
}

#[cfg(FALSE)]
fn no_leading_tuple_struct() {
    let TS( | A | B ); //~ ERROR expected pattern, found `|`
}

#[cfg(FALSE)]
fn no_leading_struct() {
    let NS { f: | A | B }; //~ ERROR expected pattern, found `|`
}
