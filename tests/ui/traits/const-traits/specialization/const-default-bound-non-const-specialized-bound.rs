// Tests that trait bounds on specializing trait impls must be `[const]` if the
// same bound is present on the default impl and is `[const]` there.

#![feature(const_trait_impl)]
#![feature(rustc_attrs)]
#![feature(min_specialization)]

#[rustc_specialization_trait]
trait Specialize {}

const trait Foo {}

const trait Bar {
    fn bar();
}

impl<T> const Bar for T
where
    T: [const] Foo,
{
    default fn bar() {}
}

impl<T> Bar for T //~ ERROR conflicting implementations of trait `Bar`
where
    T: Foo,
    T: Specialize,
{
    fn bar() {}
}

const trait Baz {
    fn baz();
}

impl<T> const Baz for T
where
    T: [const] Foo,
{
    default fn baz() {}
}

impl<T> const Baz for T //~ ERROR conflicting implementations of trait `Baz`
where
    T: Foo,
    T: Specialize,
{
    fn baz() {}
}

fn main() {}
