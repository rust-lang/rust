// Tests that `T: Foo` and `T: ~const Foo` are treated as equivalent for the
// purposes of min_specialization.

// check-pass

#![feature(rustc_attrs)]
#![feature(min_specialization)]
#![feature(const_trait_impl)]

#[rustc_specialization_trait]
trait Specialize {}

#[const_trait]
trait Foo {}

#[const_trait]
trait Bar {}

impl<T> Bar for T
where
    T: Foo,
{}

impl<T> const Bar for T
where
    T: ~const Foo,
    T: Specialize,
{}

fn main() {}
