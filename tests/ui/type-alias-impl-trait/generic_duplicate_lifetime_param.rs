#![feature(type_alias_impl_trait)]

fn main() {}

type Two<'a, 'b> = impl std::fmt::Debug;

fn one<'a>(t: &'a ()) -> Two<'a, 'a> {
    //~^ ERROR non-defining opaque type use
    t
    //~^ ERROR non-defining opaque type use
}
