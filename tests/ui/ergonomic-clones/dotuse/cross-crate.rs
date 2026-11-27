//! `.use` from a dependency with `ergonomic_clones` copies, without the feature.
//@ aux-build: cross-crate-aux.rs
//@ revisions: mono inline
//@[mono] compile-flags: -Zshare-generics=no
//@[inline] compile-flags: -O -Zinline-mir
//@ run-pass

extern crate cross_crate_aux as aux;

fn main() {
    let _ = aux::dup(aux::S);
    assert_eq!(aux::clones(), 0);
}
