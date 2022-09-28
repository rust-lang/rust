#![feature(type_alias_impl_trait)]

type Opaque<'a, T> = impl Sized;
fn defining<'a, T>(x: &'a i32) -> Opaque<T> { x }
//~^ ERROR: non-defining opaque type use in defining scope

fn main() {}
