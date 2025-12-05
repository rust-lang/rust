//@ aux-build:unstable_impl_coherence_aux.rs
//@ revisions: enabled disabled

#![cfg_attr(enabled, feature(foo))]
extern crate unstable_impl_coherence_aux as aux;
use aux::Trait;

/// Coherence test for unstable impl.
/// No matter feature `foo` is enabled or not, the impl
/// for aux::Trait will be rejected by coherence checking.

struct LocalTy;

impl aux::Trait for LocalTy {}
//~^ ERROR: conflicting implementations of trait `Trait` for type `LocalTy`

fn main(){}
