#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

// Issue 110549

pub trait TraitWAssocConst {
    #[type_const]
    const A: usize;
}

fn foo<T: TraitWAssocConst<A = 32>>() {}

fn bar<T: TraitWAssocConst>() {
    foo::<T>();
    //~^ ERROR type mismatch resolving `<T as TraitWAssocConst>::A == 32`
}

fn main() {}
