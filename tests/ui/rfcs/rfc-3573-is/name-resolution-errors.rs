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
    } else {
        x; //~ ERROR cannot find value `x` in this scope
    }
    x; //~ ERROR cannot find value `x` in this scope

    // Elsewhere, `is`'s bindings only extend to the end of its `&&`-chain.
    if x //~ ERROR cannot find value `x` in this scope
        && (is!(true is x) && x && { x })
        && x //~ ERROR cannot find value `x` in this scope
        && (is!(true is x) || x) //~ ERROR cannot find value `x` in this scope
        && (!is!(true is x) || x) //~ ERROR cannot find value `x` in this scope
        && (is!(true is x) & x) //~ ERROR cannot find value `x` in this scope
    {
        x; //~ ERROR cannot find value `x` in this scope
    }

    match is!(true is x) {
        _ => { x; } //~ ERROR cannot find value `x` in this scope
    }

    match () {
        () if is!(true is x) && x => { x; }
        () if x => {} //~ ERROR cannot find value `x` in this scope
    }

    match () {
        () if is!(true is x) && x => { x; }
        () => { x; } //~ ERROR cannot find value `x` in this scope
    }

    fn f(_a: bool, _b: bool) {}
    f(is!(true is x) && x, true);
    f(is!(true is x), x); //~ ERROR cannot find value `x` in this scope
    f(is!(true is x) && x, x); //~ ERROR cannot find value `x` in this scope

    if !is!(true is x) {
        x; //~ ERROR cannot find value `x` in this scope
    } else {
        x; //~ ERROR cannot find value `x` in this scope
    }

    if true
        && if is!(true is x) && x { x } else { false }
        && x //~ ERROR cannot find value `x` in this scope
    {
        x; //~ ERROR cannot find value `x` in this scope
    }

    while is!(true is x) {
        x;
        break;
    }
    x; //~ ERROR cannot find value `x` in this scope

    is!(true is x);
    x; //~ ERROR cannot find value `x` in this scope

    is!(true is x);
    x; //~ ERROR cannot find value `x` in this scope
}
