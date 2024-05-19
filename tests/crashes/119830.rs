//@ known-bug: #119830
#![feature(effects)]
#![feature(min_specialization)]

trait Specialize {}

trait Foo {}

impl<T> const Foo for T {}

impl<T> const Foo for T where T: const Specialize {}
