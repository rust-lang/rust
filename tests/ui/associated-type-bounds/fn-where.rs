//@ run-pass
//@ aux-build:fn-aux.rs

#![allow(unused)]
extern crate fn_aux;

use fn_aux::*;

// ATB, Type parameters, Where-clauses:

fn where_bound<B>(beta: B) -> usize
where
    B: Beta<Gamma: Alpha>
{
    desugared_bound(beta)
}

fn where_bound_region<B>(beta: B) -> usize
where
    B: Beta<Gamma: 'static>
{
    desugared_bound_region(beta)
}

fn where_bound_multi<B>(beta: B) -> usize
where
    B: Copy + Beta<Gamma: Alpha + 'static + Delta>,
{
    desugared_bound_multi(beta)
}

fn where_bound_region_specific<'a, B>(gamma: &'a B::Gamma) -> usize
where
    B: Beta<Gamma: 'a + Epsilon<'a>>,
{
    desugared_bound_region_specific::<B>(gamma)
}

fn where_bound_region_forall<B>(beta: B) -> usize
where
    B: Beta<Gamma: Copy + for<'a> Epsilon<'a>>,
{
    desugared_bound_region_forall(beta)
}

fn where_bound_region_forall2<B>(beta: B) -> usize
where
    B: Beta<Gamma: Copy + for<'a> Epsilon<'a, Zeta: Eta>>,
{
    desugared_bound_region_forall2(beta)
}

fn where_contraint_region_forall<B>(beta: B) -> usize
where
    for<'a> &'a B: Beta<Gamma: Alpha>,
{
    desugared_contraint_region_forall(beta)
}

fn where_bound_nested<B>(beta: B) -> usize
where
    B: Beta<Gamma: Copy + Alpha + Beta<Gamma: Delta>>,
{
    desugared_bound_nested(beta)
}

fn main() {
    let beta = BetaType;
    let gamma = beta.gamma();

    assert_eq!(42, where_bound(beta));
    assert_eq!(24, where_bound_region(beta));
    assert_eq!(42 + 24 + 1337, where_bound_multi(beta));
    assert_eq!(7331, where_bound_region_specific::<BetaType>(&gamma));
    assert_eq!(7331 * 2, where_bound_region_forall::<BetaType>(beta));
    assert_eq!(42 + 1337, where_bound_nested::<BetaType>(beta));
}
