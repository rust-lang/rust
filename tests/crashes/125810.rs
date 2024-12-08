//@ known-bug: rust-lang/rust#125810
#![feature(arbitrary_self_types, dispatch_from_dyn)]

use std::ops::{Deref, DispatchFromDyn};

trait Trait<T: Deref<Target = Self> + DispatchFromDyn<T>> {
    fn MONO_BUF(self: T) -> dyn Trait<T>;
}

fn main() {}
