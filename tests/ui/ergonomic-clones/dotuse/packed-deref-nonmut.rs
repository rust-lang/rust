#![feature(ergonomic_clones)]
#![expect(incomplete_features)]

use std::clone::UseCloned;
use std::ops::{Deref, Index};

#[derive(Clone)]
#[repr(align(32))]
struct Foo;

impl UseCloned for Foo {}

#[repr(packed)]
struct Packed(Foo);

struct ImplsDeref(Packed);

impl Deref for ImplsDeref {
    type Target = Packed;

    fn deref(&self) -> &Packed {
        &self.0
    }
}

impl Index<usize> for ImplsDeref {
    type Output = Packed;

    fn index(&self, _: usize) -> &Packed {
        &self.0
    }
}

fn main() {
    let mut derefable = ImplsDeref(Packed(Foo));
    let _used = (*derefable).0.use;
    //~^ ERROR cannot borrow data in dereference of `ImplsDeref` as mutable

    let mut indexable = ImplsDeref(Packed(Foo));
    let _used = indexable[0].0.use;
    //~^ ERROR cannot borrow data in an index of `ImplsDeref` as mutable
}
