// Tests that `T: ~const Foo` in a specializing impl is treated as equivalent to
// `T: Foo` in the default impl for the purposes of specialization (i.e., it
// does not think that the user is attempting to specialize on trait `Foo`).

// check-pass

#![feature(rustc_attrs)]
#![feature(min_specialization)]
#![feature(const_trait_impl)]

#[rustc_specialization_trait]
trait Specialize {}

#[const_trait]
trait Foo {}

#[const_trait]
trait Bar {
    fn bar();
}

impl<T> Bar for T
where
    T: Foo,
{
    default fn bar() {}
}

impl<T> const Bar for T
where
    T: ~const Foo,
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
    T: Foo,
{
    default fn baz() {}
}

impl<T> const Baz for T
where
    T: ~const Foo,
    T: Specialize,
{
    fn baz() {}
}

fn main() {}
