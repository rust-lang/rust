//@ revisions: normal abby
//@ compile-flags: -Znext-solver=globally
//@ [abby] compile-flags: -Zassumptions-on-binders

use std::cell::Cell;

fn covariant<'a: 'b, 'b>(x: &'b ()) -> &'a () {
    x
    //~^ ERROR lifetime may not live long enough
}

fn contravariant<'a: 'b, 'b>(x: fn(&'a ())) -> fn(&'b ()) {
    x
    //~^ ERROR lifetime may not live long enough
}

fn invariant<'a: 'b, 'b>(x: Cell<&'a ()>) -> Cell<&'b ()> {
    x
    //~^ ERROR lifetime may not live long enough
}

fn higher_ranked(x: fn(&'static ())) -> for<'a> fn(&'a ()) {
    x
    //~^ ERROR mismatched types
}

fn main() {}
