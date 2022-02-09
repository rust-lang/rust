#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

type OneLifetime<'a, 'b> = impl std::fmt::Debug;

fn foo<'a, 'b>(a: &'a u32, b: &'b u32) -> OneLifetime<'a, 'b> {
    a
}

fn bar<'a, 'b>(a: &'a u32, b: &'b u32) -> OneLifetime<'a, 'b> {
    b
    //~^ ERROR: concrete type differs from previous defining opaque type use
}

fn main() {}
