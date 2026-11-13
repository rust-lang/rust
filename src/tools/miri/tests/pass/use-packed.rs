#![feature(ergonomic_clones)]
#![expect(incomplete_features)]

use std::clone::UseCloned;
use std::ops::{Deref, DerefMut, Index, IndexMut};

#[allow(dead_code)]
#[derive(Clone)]
struct Foo(u128);

impl UseCloned for Foo {}

#[repr(packed)]
struct Packed<T>(T);

struct ImplsDerefMut(Packed<Foo>);

impl Deref for ImplsDerefMut {
    type Target = Packed<Foo>;

    fn deref(&self) -> &Packed<Foo> {
        &self.0
    }
}

impl DerefMut for ImplsDerefMut {
    fn deref_mut(&mut self) -> &mut Packed<Foo> {
        &mut self.0
    }
}

impl Index<usize> for ImplsDerefMut {
    type Output = Packed<Foo>;

    fn index(&self, _: usize) -> &Packed<Foo> {
        &self.0
    }
}

impl IndexMut<usize> for ImplsDerefMut {
    fn index_mut(&mut self, _: usize) -> &mut Packed<Foo> {
        &mut self.0
    }
}

struct ImplsDeref(Packed<u32>);

impl Deref for ImplsDeref {
    type Target = Packed<u32>;

    fn deref(&self) -> &Packed<u32> {
        &self.0
    }
}

impl Index<usize> for ImplsDeref {
    type Output = Packed<u32>;

    fn index(&self, _: usize) -> &Packed<u32> {
        &self.0
    }
}

fn main() {
    let packed = Packed(Foo(42));
    let _used = packed.0.use;
    let _used = (|| packed.0.use)();

    let mut packed = packed;
    let packed_mut = &mut packed;
    let _used = packed_mut.0.use;
    let _used = (|| packed_mut.0.use)();

    let mut derefable = ImplsDerefMut(Packed(Foo(42)));
    let _used = (*derefable).0.use;

    let mut indexable = ImplsDerefMut(Packed(Foo(42)));
    let _used = indexable[0].0.use;

    let _used = (*ImplsDeref(Packed(42))).0.use;
    let _used = ImplsDeref(Packed(42))[0].0.use;
}
