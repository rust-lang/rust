//@ known-bug: #57276

#![feature(arbitrary_self_types, dispatch_from_dyn)]

use std::ops::{Deref, DispatchFromDyn};

trait Trait<T: Deref<Target = Self> + DispatchFromDyn<T>> {
    fn foo(self: T) -> dyn Trait<T>;
}

fn main() {}
