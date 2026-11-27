#![feature(type_alias_impl_trait)]
#![allow(dead_code)]

type OneLifetime<'a, 'b> = impl std::fmt::Debug;

#[define_opaque(OneLifetime)]
fn foo<'a, 'b>(a: &'a u32, b: &'b u32) -> OneLifetime<'a, 'b> {
    a
}

#[define_opaque(OneLifetime)]
fn bar<'a, 'b>(a: &'a u32, b: &'b u32) -> OneLifetime<'a, 'b> {
    b
    //~^ ERROR: concrete type differs from previous defining opaque type use
}

fn main() {}
