//! Use items with GCE where-clauses from a dependency, without the feature.
//@ aux-build: dependent-crate-no-gce-where-clause-aux.rs
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver
//@ build-pass

extern crate dependent_crate_no_gce_where_clause_aux as aux;

fn needs_send<T: Send>(_: &T) {}

struct L;

impl aux::Tr<2> for L {
    fn g(&self) -> [u8; 3] {
        [0; 3]
    }
}

fn main() {
    let _ = aux::f::<3>();
    let w = aux::mk::<2>();
    needs_send(&w);
    let _ = aux::Tr::g(&L);
}
