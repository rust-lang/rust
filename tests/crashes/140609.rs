//@ known-bug: #140609
#![feature(with_negative_coherence)]
#![feature(generic_const_exprs)]
#![crate_type = "lib"]
trait Trait {}
struct A<const B: bool>;

trait C {}

impl<const D: u32> Trait for E<D> where A<{ D <= 2 }>: FnOnce(&isize) {}
struct E<const D: u32>;

impl<const D: u32> Trait for E<D> where A<{ D <= 2 }>: C {}
