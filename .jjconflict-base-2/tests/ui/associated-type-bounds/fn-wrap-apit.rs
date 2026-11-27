//@ run-pass
//@ aux-build:fn-aux.rs

#![allow(dead_code)]

extern crate fn_aux;

use fn_aux::*;

// ATB, APIT + Wrap:

struct Wrap<T>(T);

fn wrap_apit_bound(beta: Wrap<impl Beta<Gamma: Alpha>>) -> usize {
    desugared_bound(beta.0)
}

fn wrap_apit_bound_region(beta: Wrap<impl Beta<Gamma: 'static>>) -> usize {
    desugared_bound_region(beta.0)
}

fn wrap_apit_bound_multi(
    beta: Wrap<impl Copy + Beta<Gamma: Alpha + 'static + Delta>>
) -> usize {
    desugared_bound_multi(beta.0)
}

fn wrap_apit_bound_region_forall(
    beta: Wrap<impl Beta<Gamma: Copy + for<'a> Epsilon<'a>>>
) -> usize {
    desugared_bound_region_forall(beta.0)
}

fn wrap_apit_bound_region_forall2(
    beta: Wrap<impl Beta<Gamma: Copy + for<'a> Epsilon<'a, Zeta: Eta>>>
) -> usize {
    desugared_bound_region_forall2(beta.0)
}

fn wrap_apit_bound_nested(
    beta: Wrap<impl Beta<Gamma: Copy + Alpha + Beta<Gamma: Delta>>>
) -> usize {
    desugared_bound_nested(beta.0)
}

fn wrap_apit_bound_nested2(
    beta: Wrap<impl Beta<Gamma = impl Copy + Alpha + Beta<Gamma: Delta>>>
) -> usize {
    desugared_bound_nested(beta.0)
}

fn main() {
    let beta = BetaType;
    let _gamma = beta.gamma();

    assert_eq!(42, wrap_apit_bound(Wrap(beta)));
    assert_eq!(24, wrap_apit_bound_region(Wrap(beta)));
    assert_eq!(42 + 24 + 1337, wrap_apit_bound_multi(Wrap(beta)));
    assert_eq!(7331 * 2, wrap_apit_bound_region_forall(Wrap(beta)));
    // FIXME: requires lazy normalization.
    // assert_eq!(7331 * 2, wrap_apit_bound_region_forall2(Wrap(beta)));
    assert_eq!(42 + 1337, wrap_apit_bound_nested(Wrap(beta)));
    assert_eq!(42 + 1337, wrap_apit_bound_nested2(Wrap(beta)));
}
