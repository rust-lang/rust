//@ run-pass
//@ aux-build:fn-aux.rs

#![allow(unused)]
extern crate fn_aux;

use fn_aux::*;

// ATB, Type parameters, Inline bounds:

fn inline_bound<B: Beta<Gamma: Alpha>>(beta: B) -> usize {
    desugared_bound(beta)
}

fn inline_bound_region<B: Beta<Gamma: 'static>>(beta: B) -> usize {
    desugared_bound_region(beta)
}

fn inline_bound_multi<B: Copy + Beta<Gamma: Alpha + 'static + Delta>>(
    beta: B
) -> usize {
    desugared_bound_multi(beta)
}

fn inline_bound_region_specific<'a, B: Beta<Gamma: 'a + Epsilon<'a>>>(
    gamma: &'a B::Gamma
) -> usize {
    desugared_bound_region_specific::<B>(gamma)
}

fn inline_bound_region_forall<B: Beta<Gamma: Copy + for<'a> Epsilon<'a>>>(
    beta: B
) -> usize {
    desugared_bound_region_forall(beta)
}

fn inline_bound_region_forall2<B: Beta<Gamma: Copy + for<'a> Epsilon<'a, Zeta: Eta>>>(
    beta: B
) -> usize {
    desugared_bound_region_forall2(beta)
}

fn inline_bound_nested<B: Beta<Gamma: Copy + Alpha + Beta<Gamma: Delta>>>(
    beta: B
) -> usize {
    desugared_bound_nested(beta)
}

fn main() {
    let beta = BetaType;
    let gamma = beta.gamma();

    assert_eq!(42, inline_bound(beta));
    assert_eq!(24, inline_bound_region(beta));
    assert_eq!(42 + 24 + 1337, inline_bound_multi(beta));
    assert_eq!(7331, inline_bound_region_specific::<BetaType>(&gamma));
    assert_eq!(7331 * 2, inline_bound_region_forall(beta));
    // FIXME: requires lazy normalization.
    // assert_eq!(7331 * 2, inline_bound_region_forall2(beta));
    assert_eq!(42 + 1337, inline_bound_nested(beta));
}
