#![feature(arbitrary_self_types, dispatch_from_dyn)]

use std::ops::{Deref, DispatchFromDyn};

trait Trait<T: Deref<Target=Self> + DispatchFromDyn<T>> {
    fn foo(self: T) -> dyn Trait<T>;
    //~^ ERROR: associated item referring to unboxed trait object for its own trait
    //~| ERROR: the trait `Trait` is not dyn compatible
}

fn main() {}
