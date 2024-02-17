//@ revisions: stock gce

#![feature(associated_const_equality)]
#![cfg_attr(gce, feature(generic_const_exprs))]
//[gce]~^ WARN the feature `generic_const_exprs` is incomplete

trait TraitWAssocConst {
    const A: usize;
}

fn foo<T: TraitWAssocConst<A = 1>>() {}

fn bar<T: TraitWAssocConst<A = 0>>() {
    foo::<T>();
    //~^ ERROR type mismatch resolving `<T as TraitWAssocConst>::A == 1`
}

fn main() {}
