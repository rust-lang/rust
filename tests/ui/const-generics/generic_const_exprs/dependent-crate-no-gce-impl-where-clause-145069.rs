//! Regression test for <https://github.com/rust-lang/rust/issues/145069>.
//@ aux-build: dependent-crate-no-gce-impl-where-clause-aux.rs
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver
//@ run-pass

extern crate dependent_crate_no_gce_impl_where_clause_aux as aux;

use aux::{Buggy, Conf};

struct UserStruct;

impl Conf for UserStruct {
    const LAYER_COUNT: usize = 0;
}

fn main() {
    assert_eq!(UserStruct::TEST, 0);
}
