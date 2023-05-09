// Tests that trait bounds on specializing trait impls must be `~const` if the
// same bound is present on the default impl and is `~const` there.

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

// bgr360: I was only able to exercise the code path that raises the
// "missing ~const qualifier" error by making this base impl non-const, even
// though that doesn't really make sense to do. As seen below, if the base impl
// is made const, rustc fails earlier with an overlapping impl failure.
impl<T> Bar for T
where
    T: ~const Foo,
{
    default fn bar() {}
}

impl<T> Bar for T
where
    T: Foo, //~ ERROR missing `~const` qualifier
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
    T: ~const Foo,
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
