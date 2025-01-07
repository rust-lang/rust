// Test that we do not attempt to create dyn-incompatible trait objects in const eval.

//@ revisions: curr dyn_compatible_for_dispatch

#![cfg_attr(dyn_compatible_for_dispatch, feature(dyn_compatible_for_dispatch))]

trait Qux {
    fn bar();
}

static FOO: &(dyn Qux + Sync) = "desc";
//~^ the trait `Qux` cannot be made into an object
//[curr]~| the trait `Qux` cannot be made into an object
//[curr]~| the trait `Qux` cannot be made into an object

fn main() {}
