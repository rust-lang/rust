//@ revisions: current next
//@[next] compile-flags: -Znext-solver

#![feature(trait_upcasting)]

trait Super {
    type Assoc;
}

trait Sub: Super {}

impl<T: ?Sized> Super for T {
    type Assoc = i32;
}

fn illegal(x: &dyn Sub<Assoc = ()>) -> &dyn Super<Assoc = i32> { x }
//~^ ERROR mismatched types

// Want to make sure that we can't "upcast" to a supertrait that has a different
// associated type that is instead provided by a blanket impl (and doesn't come
// from the object bounds).

fn main() {}
