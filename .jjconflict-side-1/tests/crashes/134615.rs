//@ known-bug: #134615

#![feature(generic_const_exprs)]

trait Trait {
    const CONST: usize;
}

fn f()
where
    for<'a> (): Trait,
    [(); <() as Trait>::CONST]:,
{
}

pub fn main() {}
