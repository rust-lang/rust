#![feature(ergonomic_clones)]
#![expect(incomplete_features)]

use std::clone::UseCloned;
use std::ops::{Deref, DerefMut, Index, IndexMut};

#[derive(Clone)]
#[repr(align(32))]
struct Foo;

impl UseCloned for Foo {}

#[repr(packed)]
struct Packed(Foo);

struct ImplsDerefMut(Packed);

impl Deref for ImplsDerefMut {
    type Target = Packed;

    fn deref(&self) -> &Packed {
        &self.0
    }
}

impl DerefMut for ImplsDerefMut {
    fn deref_mut(&mut self) -> &mut Packed {
        &mut self.0
    }
}

impl Index<usize> for ImplsDerefMut {
    type Output = Packed;

    fn index(&self, _: usize) -> &Packed {
        &self.0
    }
}

impl IndexMut<usize> for ImplsDerefMut {
    fn index_mut(&mut self, _: usize) -> &mut Packed {
        &mut self.0
    }
}

fn main() {
    let derefable = ImplsDerefMut(Packed(Foo));
    let _used = (*derefable).0.use;
    //~^ ERROR cannot borrow `derefable` as mutable, as it is not declared as mutable

    let indexable = ImplsDerefMut(Packed(Foo));
    let _used = indexable[0].0.use;
    //~^ ERROR cannot borrow `indexable` as mutable, as it is not declared as mutable
}
