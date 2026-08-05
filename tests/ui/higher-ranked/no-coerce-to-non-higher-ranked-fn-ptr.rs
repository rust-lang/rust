// higher-ranked to non-higher-ranked function pointer coercion applies only
// to the outermost binder, at a coercion site. It must not fire for nested
// binders, nor in the reverse direction.

use std::cell::Cell;

// The coercion does not recurse into type constructors.
fn nested(x: Vec<for<'a> fn(&'a ())>) -> Vec<fn(&'static ())> {
    x
    //~^ ERROR mismatched types
}

// a non-higher-ranked fn pointer is not more general higher-ranked fn pointer.
fn reverse(x: fn(&'static ())) -> for<'a> fn(&'a ()) {
    x
    //~^ ERROR mismatched types
}

// Nested inside an invariant constructor is equality, and these are not equal.
fn invariant(x: Cell<for<'a> fn(&'a ())>) -> Cell<fn(&'static ())> {
    x
    //~^ ERROR mismatched types
}

fn main() {}
