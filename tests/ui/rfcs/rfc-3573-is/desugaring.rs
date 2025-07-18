//! Test desugaring of `is` expressions to `let` and `if`.
//@ check-pass
//@ compile-flags: -Z unpretty=hir
//@ edition: 2024
#![feature(builtin_syntax)]

fn main() {
    // At the top level of an `if` or `while` condition, `is` desugars directly to `let`.
    if true && builtin # is(0 is 0) && true {}
    while true && builtin # is(0 is 0) && true {}

    // Otherwise, an `&&`-chain with `is` in it is wrapped in an `if` expression.
    builtin # is(0 is 0);
    true && builtin # is(0 is 0) && true;

    // `let` isn't allowed under parentheses or other operators.
    // `is` anywhere other than the top level `&&`-chain of a condition is wrapped in an `if`.
    if (builtin # is(0 is 0)) {}
    if (true && builtin # is(0 is 0) && true) {}
    if builtin # is(0 is 0) || true && builtin # is(0 is 0) && true {}
}
