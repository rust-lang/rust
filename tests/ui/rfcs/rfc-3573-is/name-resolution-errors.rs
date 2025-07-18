//! Test that the lexical scope for `is`'s bindings doesn't extend farther than it should.
//@ edition: 2024
//@ aux-crate: is_macro=is-macro.rs
#![feature(builtin_syntax)]
use is_macro::is;

fn main() {
    // `is` used where `let` expressions can be used behaves like a `let` expression.
    if x //~ ERROR cannot find value `x` in this scope
        && is!(true is x)
        && x
    {
        x;
    }
    x; //~ ERROR cannot find value `x` in this scope

    // Elsewhere, `is`'s bindings only extend to the end of its `&&`-chain.
    if x //~ ERROR cannot find value `x` in this scope
        && (is!(true is x) && x)
        && x //~ ERROR cannot find value `x` in this scope
        && (is!(true is x) || x) //~ ERROR cannot find value `x` in this scope
    {
        x; //~ ERROR cannot find value `x` in this scope
    }
    is!(true is x);
    x; //~ ERROR cannot find value `x` in this scope
}
