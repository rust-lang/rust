//! Normalize GCE consts from a dependency in typeck, without the feature.
//@ aux-build: dependent-crate-no-gce-typeck-aux.rs
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver
//@ run-pass

extern crate dependent_crate_no_gce_typeck_aux as aux;

use aux::{Foo, mk};

fn main() {
    let v = vec![mk(Foo::<3>)];
    assert_eq!(v[0].len(), 4);
    match mk(Foo::<3>) {
        [0, .., 0] => {}
        _ => panic!(),
    }
}
