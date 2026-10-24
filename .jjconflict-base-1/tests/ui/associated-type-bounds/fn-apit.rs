//@ run-pass
//@ aux-build:fn-aux.rs

#![allow(unused)]
extern crate fn_aux;

use fn_aux::*;

fn apit_bound(beta: impl Beta<Gamma: Alpha>) -> usize {
    desugared_bound(beta)
}

fn apit_bound_region(beta: impl Beta<Gamma: 'static>) -> usize {
    desugared_bound_region(beta)
}

fn apit_bound_multi(
    beta: impl Copy + Beta<Gamma: Alpha + 'static + Delta>
) -> usize {
    desugared_bound_multi(beta)
}

fn apit_bound_region_forall(
    beta: impl Beta<Gamma: Copy + for<'a> Epsilon<'a>>
) -> usize {
    desugared_bound_region_forall(beta)
}

fn apit_bound_region_forall2(
    beta: impl Beta<Gamma: Copy + for<'a> Epsilon<'a, Zeta: Eta>>
) -> usize {
    desugared_bound_region_forall2(beta)
}

fn apit_bound_nested(
    beta: impl Beta<Gamma: Copy + Alpha + Beta<Gamma: Delta>>
) -> usize {
    desugared_bound_nested(beta)
}

fn apit_bound_nested2(
    beta: impl Beta<Gamma = impl Copy + Alpha + Beta<Gamma: Delta>>
) -> usize {
    desugared_bound_nested(beta)
}

fn main() {
    let beta = BetaType;
    let _gamma = beta.gamma();

    assert_eq!(42, apit_bound(beta));
    assert_eq!(24, apit_bound_region(beta));
    assert_eq!(42 + 24 + 1337, apit_bound_multi(beta));
    assert_eq!(7331 * 2, apit_bound_region_forall(beta));
    assert_eq!(42 + 1337, apit_bound_nested(beta));
    assert_eq!(42 + 1337, apit_bound_nested2(beta));
}
