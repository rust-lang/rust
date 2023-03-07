#![feature(type_alias_impl_trait)]

pub trait Captures<'a> {}

impl<'a, T: ?Sized> Captures<'a> for T {}

type X<'a, 'b> = impl std::fmt::Debug + Captures<'a> + Captures<'b>;

fn f<'t, 'u>(a: &'t u32, b: &'u u32) -> (X<'t, 'u>, X<'u, 't>) {
    (a, a)
    //~^ ERROR concrete type differs from previous defining opaque type use
}

fn main() {}
