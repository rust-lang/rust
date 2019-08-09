// Test that Cell is considered invariant with respect to its
// type.

#![feature(rustc_attrs)]

use std::cell::Cell;

// For better or worse, associated types are invariant, and hence we
// get an invariant result for `'a`.
#[rustc_variance]
struct Foo<'a> { //~ ERROR [o]
    x: Box<dyn Fn(i32) -> &'a i32 + 'static>
}

fn main() {
}
