//@ revisions: normal abby
//@ check-pass
//@ compile-flags: -Znext-solver=globally
//@ [abby] compile-flags: -Zassumptions-on-binders

use std::cell::Cell;

fn covariant<'a: 'b, 'b>(x: &'a ()) -> &'b () {
    x
}

fn contravariant<'a: 'b, 'b>(x: fn(&'b ())) -> fn(&'a ()) {
    x
}

fn invariant<'a: 'b, 'b: 'a>(x: Cell<&'a ()>) -> Cell<&'b ()> {
    x
}

fn nested<'a: 'b, 'b>(x: for<'c> fn(&'b &'c ())) -> for<'c> fn(&'a &'c ()) {
    x
}

fn higher_ranked(x: for<'a> fn(&'a ())) -> fn(&'static ()) {
    x
}

fn main() {}
