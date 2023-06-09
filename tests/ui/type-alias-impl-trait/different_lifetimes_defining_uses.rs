#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

pub trait Captures<'a> {}

impl<'a, T: ?Sized> Captures<'a> for T {}

type OneLifetime<'a, 'b> = impl std::fmt::Debug + Captures<'a> + Captures<'b>;

fn foo<'a, 'b>(a: &'a u32, b: &'b u32) -> OneLifetime<'a, 'b> {
    a
}

fn bar<'a, 'b>(a: &'a u32, b: &'b u32) -> OneLifetime<'a, 'b> {
    b
    //~^ ERROR: concrete type differs from previous defining opaque type use
}

fn main() {}
