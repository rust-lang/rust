// Tests that `[const]` trait bounds can be used to specialize const trait impls.
// cc #95186

//@ check-pass

#![feature(const_trait_impl)]
#![feature(rustc_attrs)]
#![feature(min_specialization)]

#[rustc_specialization_trait]
const trait Specialize {}

const trait Foo {
    fn foo();
}

const impl<T> Foo for T {
    default fn foo() {}
}

const impl<T> Foo for T
where
    T: [const] Specialize,
{
    fn foo() {}
}

const trait Bar {
    fn bar() {}
}

const impl<T> Bar for T
where
    T: [const] Foo,
{
    default fn bar() {}
}

const impl<T> Bar for T
where
    T: [const] Foo,
    T: [const] Specialize,
{
    fn bar() {}
}

fn main() {}
