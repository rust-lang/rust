#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub trait TraitWAssocConst {
    const A: dyn TraitWAssocConst<A=0>; //~ ERROR: associated const equality is incomplete
    //~^ ERROR: cycle detected when computing type
}

fn main<A: TraitWAssocConst<A=0>>() {}
//~^ ERROR: associated const equality is incomplete
