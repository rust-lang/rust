// Tests that `~const` trait bounds can be used to specialize const trait impls.

// check-pass

#![feature(const_trait_impl)]
#![feature(rustc_attrs)]
#![feature(min_specialization)]

#[const_trait]
#[rustc_specialization_trait]
trait Specialize {}

#[const_trait]
trait Foo {
    fn foo();
}

impl<T> const Foo for T {
    default fn foo() {}
}

impl<T> const Foo for T
where
    T: ~const Specialize,
{
    fn foo() {}
}

#[const_trait]
trait Bar {
    fn bar() {}
}

impl<T> const Bar for T
where
    T: ~const Foo,
{
    default fn bar() {}
}

impl<T> const Bar for T
where
    T: ~const Foo,
    T: ~const Specialize,
{
    fn bar() {}
}

fn main() {}
