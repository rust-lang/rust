// Regression test for #81218
//
//@ check-pass

#![forbid(warnings)]

#[allow(unused_variables)]
fn main() {
    // We want to ensure that you don't get an error
    // here. The idea is that a derive might generate
    // code that would otherwise trigger the "unused variables"
    // lint, but it is meant to be suppressed.
    let x: ();
}
