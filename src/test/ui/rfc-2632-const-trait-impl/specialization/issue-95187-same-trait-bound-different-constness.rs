// Tests that `T: ~const Foo` and `T: Foo` are treated as equivalent for the
// purposes of min_specialization.

// check-pass

#![feature(rustc_attrs)]
#![feature(min_specialization)]
#![feature(const_trait_impl)]

#[rustc_specialization_trait]
trait Specialize {}

trait Foo {}

trait Bar {}

impl<T> const Bar for T
where
    T: ~const Foo,
{}

impl<T> Bar for T
where
    T: Foo,
    T: Specialize,
{}

fn main() {}
