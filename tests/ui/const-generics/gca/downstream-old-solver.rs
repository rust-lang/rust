//! Report, instead of ICEing on, gca consts from a dependency in the old solver.
//@ aux-build: gca-const-items-upstream-166244.rs
//@ compile-flags: -Znext-solver=coherence -Zdeduplicate-diagnostics=yes

extern crate gca_const_items_upstream_166244 as up;

fn needs_send<T: Send>(_: &T) {}

fn g<const N: usize>(w: up::W<N>) -> usize {
    needs_send(&w);
    w.0.len()
}

fn main() {
    let _ = g::<3>(up::mk::<3>());
}

//~? ERROR `ADD1` is from a crate with `gca_const_items`
