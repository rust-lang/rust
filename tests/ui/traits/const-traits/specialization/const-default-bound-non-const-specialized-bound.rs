// Tests that trait bounds on specializing trait impls must be `[const]` if the
// same bound is present on the default impl and is `[const]` there.
//@ known-bug: #110395
// FIXME(const_trait_impl) ^ should error

#![feature(const_trait_impl)]
#![feature(rustc_attrs)]
#![feature(min_specialization)]

#[rustc_specialization_trait]
trait Specialize {}

#[const_trait]
trait Foo {}

#[const_trait]
trait Bar {
    fn bar();
}

impl<T> const Bar for T
where
    T: [const] Foo,
{
    default fn bar() {}
}

impl<T> Bar for T
where
    T: Foo, //FIXME ~ ERROR missing `[const]` qualifier
    T: Specialize,
{
    fn bar() {}
}

#[const_trait]
trait Baz {
    fn baz();
}

impl<T> const Baz for T
where
    T: [const] Foo,
{
    default fn baz() {}
}

impl<T> const Baz for T //FIXME ~ ERROR conflicting implementations of trait `Baz`
where
    T: Foo,
    T: Specialize,
{
    fn baz() {}
}

fn main() {}
