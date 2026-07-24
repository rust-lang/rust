//@ check-pass

// A higher-ranked function pointer `for<'a> fn(&'a ())` still coerces to
// `fn(&'static ())`, but only at a coercion site (a return, a `let`, or a call
// argument). Separately, alpha-equivalent higher-ranked types
// must still relate by equality through an invariant constructor.

use std::cell::Cell;

fn at_return(x: for<'a> fn(&'a ())) -> fn(&'static ()) {
    x
}

fn at_let() {
    let x: for<'a> fn(&'a ()) = |_| ();
    let _y: fn(&'static ()) = x;
}

fn requires_non_hr<'a, F: FnOnce(&'a ())>(_: F) {}
fn at_call() {
    let x: for<'hr> fn(&'hr ()) = |_| ();
    requires_non_hr(x);
}

fn alpha_equivalent(x: Cell<for<'a> fn(&'a ())>) -> Cell<for<'b> fn(&'b ())> {
    x
}

fn main() {}
