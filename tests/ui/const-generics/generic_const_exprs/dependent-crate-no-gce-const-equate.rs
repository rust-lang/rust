//! Relate GCE consts from a dependency, without the feature.
//@ aux-build: dependent-crate-no-gce-const-equate-aux.rs
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver
//@ run-pass

extern crate dependent_crate_no_gce_const_equate_aux as aux;

use aux::{Foo, Tr};

struct L;

impl Tr<4> for L {}

fn needs<T: Tr<4>>(t: T) -> usize {
    t.v()
}

fn main() {
    assert_eq!(needs(Foo::<3>), 4);
    assert_eq!(needs(L), 4);
    assert_eq!(Foo::<3>.v(), 4);
    assert_eq!(<Foo<5> as Tr<_>>::v(&Foo), 6);
    let a: [u8; 4] = aux::g([0; 3]);
    assert_eq!(a.len(), 4);
}
