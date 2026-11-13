//@ aux-build:hidden-child.rs

// Regression test for #153477.
// When a re-export is #[doc(hidden)], diagnostics should prefer
// the canonical path (e.g. `Some`) over the hidden re-export path
// (e.g. `hidden_child::__private::Some`).

extern crate hidden_child;

fn main() {
    let x: Option<i32> = 1i32; //~ ERROR mismatched types
}
