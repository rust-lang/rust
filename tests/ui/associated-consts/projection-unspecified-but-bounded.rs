#![feature(associated_const_equality)]

// Issue 110549

pub trait TraitWAssocConst {
    const A: usize;
}

fn foo<T: TraitWAssocConst<A = 32>>() {}

fn bar<T: TraitWAssocConst>() {
    foo::<T>();
    //~^ ERROR type mismatch resolving `<T as TraitWAssocConst>::A == 32`
}

fn main() {}
