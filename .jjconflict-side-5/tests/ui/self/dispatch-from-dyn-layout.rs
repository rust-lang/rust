//@ check-pass
// Regression test for #57276.

// Make sure that object safety checking doesn't freak out when
// we have impossible-to-satisfy `DispatchFromDyn` predicates.

#![feature(arbitrary_self_types, dispatch_from_dyn)]

use std::ops::{Deref, DispatchFromDyn};

trait Trait<T: Deref<Target = Self> + DispatchFromDyn<T>> {
    fn foo(self: T) -> dyn Trait<T>;
}

fn main() {}
