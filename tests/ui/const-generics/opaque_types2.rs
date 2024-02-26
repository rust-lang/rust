#![feature(type_alias_impl_trait)]

type Foo = impl Sized;
//~^ ERROR: cycle

fn foo<const C: u32>() {}

const C: Foo = 42;

fn bar()
where
    Foo:,
{
    foo::<C>();
}

fn main() {}
