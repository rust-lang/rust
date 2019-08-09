// run-pass
// aux-build:fn-dyn-aux.rs

#![feature(associated_type_bounds)]

extern crate fn_dyn_aux;

use fn_dyn_aux::*;

// ATB, APIT (dyn trait):

fn dyn_apit_bound(beta: &dyn Beta<Gamma: Alpha>) -> usize {
    desugared_bound(beta)
}

fn dyn_apit_bound_region(beta: &dyn Beta<Gamma: 'static>) -> usize {
    desugared_bound_region(beta)
}

fn dyn_apit_bound_multi(
    beta: &(dyn Beta<Gamma: Alpha + 'static + Delta> + Send)
) -> usize {
    desugared_bound_multi(beta)
}

fn dyn_apit_bound_region_forall(
    beta: &dyn Beta<Gamma: Copy + for<'a> Epsilon<'a>>
) -> usize {
    desugared_bound_region_forall(beta)
}

fn dyn_apit_bound_region_forall2(
    beta: &dyn Beta<Gamma: Copy + for<'a> Epsilon<'a, Zeta: Eta>>
) -> usize {
    desugared_bound_region_forall2(beta)
}

fn dyn_apit_bound_nested(
    beta: &dyn Beta<Gamma: Copy + Alpha + Beta<Gamma: Delta>>
) -> usize {
    desugared_bound_nested(beta)
}

fn dyn_apit_bound_nested2(
    beta: &dyn Beta<Gamma = impl Copy + Alpha + Beta<Gamma: Delta>>
) -> usize {
    desugared_bound_nested(beta)
}

fn main() {
    let beta = BetaType;
    let _gamma = beta.gamma();

    assert_eq!(42, dyn_apit_bound(&beta));
    assert_eq!(24, dyn_apit_bound_region(&beta));
    assert_eq!(42 + 24 + 1337, dyn_apit_bound_multi(&beta));
    assert_eq!(7331 * 2, dyn_apit_bound_region_forall(&beta));
    assert_eq!(42 + 1337, dyn_apit_bound_nested(&beta));
    assert_eq!(42 + 1337, dyn_apit_bound_nested2(&beta));
}
