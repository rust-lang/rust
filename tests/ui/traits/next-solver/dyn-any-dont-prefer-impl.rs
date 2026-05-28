//@ compile-flags: -Znext-solver
//@ run-pass

// Test that selection prefers the builtin trait object impl for `Any`
// instead of the user defined impl. Both impls apply to the trait
// object.

use std::any::Any;

fn needs_usize(_: &usize) {}

fn main() {
    let x: &dyn Any = &1usize;
    if let Some(x) = x.downcast_ref::<usize>() {
        needs_usize(x);
    }
}
