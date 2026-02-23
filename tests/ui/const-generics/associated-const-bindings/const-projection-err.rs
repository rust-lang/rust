#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

trait TraitWAssocConst {
    type const A: usize;
}

fn foo<T: TraitWAssocConst<A = 1>>() {}

fn bar<T: TraitWAssocConst<A = 0>>() {
    foo::<T>();
    //~^ ERROR type mismatch resolving `<T as TraitWAssocConst>::A == 1`
}

fn main() {}
