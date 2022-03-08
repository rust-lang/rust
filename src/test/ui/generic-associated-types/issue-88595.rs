#![feature(generic_associated_types)]
#![feature(type_alias_impl_trait)]

fn main() {}

trait A<'a> {
    type B<'b>: Clone
    // FIXME(generic_associated_types): Remove one of the below bounds
    // https://github.com/rust-lang/rust/pull/90678#discussion_r744976085
    where
        Self: 'a, Self: 'b;

    fn a(&'a self) -> Self::B<'a>;
}

struct C;

impl<'a> A<'a> for C {
    type B<'b> = impl Clone;
    //~^ ERROR: could not find defining uses

    fn a(&'a self) -> Self::B<'a> {} //~ ERROR: non-defining opaque type use in defining scope
}
