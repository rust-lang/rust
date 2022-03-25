// Tests that `~const` trait bounds can be used to specialize const trait impls.

// check-pass

#![feature(const_trait_impl)]
#![feature(rustc_attrs)]
#![feature(min_specialization)]

#[rustc_specialization_trait]
trait Specialize {}

trait Foo {}

impl<T> const Foo for T {}

impl<T> const Foo for T
where
    T: ~const Specialize,
{}

trait Bar {}

impl<T> const Bar for T
where
    T: ~const Foo,
{}

impl<T> const Bar for T
where
    T: ~const Foo,
    T: ~const Specialize,
{}

fn main() {}
