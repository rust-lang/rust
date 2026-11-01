//@ aux-build: dyn-associated-const-cross-crate-aux-issue-311.rs
//@ compile-flags: -Znext-solver
//@ run-pass

// An unambiguous associated constant must still work when instantiated
// across crate boundaries.
extern crate dyn_associated_const_cross_crate_aux_issue_311 as aux;

fn main() {
    assert_eq!(aux::Adapter::<aux::Matching>::SIZE, 1);
    assert_eq!(aux::generic::<u8>(), 1);
    assert_eq!(aux::generic::<u16>(), 1);
    assert_eq!(aux::explicit_projection_size(), 1);
}
