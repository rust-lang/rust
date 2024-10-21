#![feature(type_alias_impl_trait)]

type Foo = impl Sized;
//~^ ERROR: cycle detected when computing type of `Foo::{opaque#0}`

fn foo<const C: u32>() {}

const C: Foo = 42;

fn bar()
where
    Foo:,
{
    foo::<C>();
}

fn main() {}
