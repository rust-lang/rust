// run-pass
// compile-flags: -Z verbose

#![allow(unused)]
#![feature(associated_type_bounds)]

pub trait Beta {
    type Gamma;
}

pub trait Delta {
}

pub trait Epsilon<'e> {
    type Zeta;
}

pub trait Eta {
}

fn where_bound_region_forall2<B>(beta: B) -> usize
where
    B: Beta<Gamma: for<'a> Epsilon<'a, Zeta: Eta>>,
{
    desugared_bound_region_forall2(beta)
}

pub fn desugared_bound_region_forall2<B>(beta: B) -> usize
where
    B: Beta,
    B::Gamma: for<'a> Epsilon<'a>,
    for<'a> <B::Gamma as Epsilon<'a>>::Zeta: Eta,
{
    0
}

fn main() {}
