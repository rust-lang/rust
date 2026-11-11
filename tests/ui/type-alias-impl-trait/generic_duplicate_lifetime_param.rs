#![feature(type_alias_impl_trait)]

fn main() {}

type Two<'a, 'b> = impl std::fmt::Debug;

#[define_opaque(Two)]
fn one<'a>(t: &'a ()) -> Two<'a, 'a> {
    t
    //~^ ERROR non-defining opaque type use
}
