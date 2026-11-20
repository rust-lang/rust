// Regression test for issue #156137.
// Computing the `SizeSkeleton` of a type whose layout depends on itself
// through a normalizing type alias used to recurse without bound and
// blow the stack. We now bail out via the recursion limit and emit a
// regular error instead of ICE-ing.

use std::mem::transmute;

trait Trait {
    type Assoc;
}
impl<T> Trait for T {
    type Assoc = T;
}
type Alias<T> = <T as Trait>::Assoc;

pub struct Thing<T: ?Sized>(*const T, Alias<Thing<T>>);

pub fn weird<T: ?Sized>(value: Thing<T>) {
    let _: i32 = unsafe { transmute(value) };
    //~^ ERROR reached the recursion limit while computing the size of `Thing<T>`
    //~| ERROR cannot transmute between types of different sizes, or dependently-sized types
}

fn main() {}
