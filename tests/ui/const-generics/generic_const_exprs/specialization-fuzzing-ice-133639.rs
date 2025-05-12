//@ check-pass

// Regression test for #133639.

#![feature(with_negative_coherence)]
#![feature(min_specialization)]
#![feature(generic_const_exprs)]
//~^ WARNING the feature `generic_const_exprs` is incomplete

#![crate_type = "lib"]
trait Trait {}
struct A<const B: bool>;

trait C {}

impl<const D: u32> Trait for E<D> where A<{ D <= 2 }>: C {}
struct E<const D: u32>;

impl<const D: u32> Trait for E<D> where A<{ D <= 2 }>: C {}
