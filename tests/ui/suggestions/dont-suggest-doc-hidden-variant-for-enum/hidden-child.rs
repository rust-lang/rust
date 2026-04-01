//@ aux-build:hidden-child.rs

// FIXME(compiler-errors): This currently suggests the wrong thing.
// UI test exists to track the problem.

extern crate hidden_child;

fn main() {
    let x: Option<i32> = 1i32; //~ ERROR mismatched types
}
