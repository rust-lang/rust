#![feature(type_alias_impl_trait)]

fn main() {}

pub trait Captures<'a> {}

impl<'a, T: ?Sized> Captures<'a> for T {}

type Two<'a, 'b> = impl std::fmt::Debug + Captures<'a> + Captures<'b>;

fn one<'a>(t: &'a ()) -> Two<'a, 'a> {
    //~^ ERROR non-defining opaque type use
    t
    //~^ ERROR non-defining opaque type use
}
