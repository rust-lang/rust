//@ run-pass
#![feature(ergonomic_clones)]
#![expect(incomplete_features)]

use std::clone::UseCloned;
use std::ops::{Deref, Index};

#[derive(Clone)]
#[repr(align(32))]
struct Foo;

impl UseCloned for Foo {}

struct NonPacked(Foo);

struct ImplsDeref(NonPacked);

impl Deref for ImplsDeref {
    type Target = NonPacked;

    fn deref(&self) -> &NonPacked {
        &self.0
    }
}

impl Index<usize> for ImplsDeref {
    type Output = NonPacked;

    fn index(&self, _: usize) -> &NonPacked {
        &self.0
    }
}

fn main() {
    let derefable = ImplsDeref(NonPacked(Foo));
    let _used = (*derefable).0.use;

    let indexable = ImplsDeref(NonPacked(Foo));
    let _used = indexable[0].0.use;
}
