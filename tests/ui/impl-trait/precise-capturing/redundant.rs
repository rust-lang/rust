//@ edition: 2024

#![deny(impl_trait_redundant_captures)]

fn hello<'a>() -> impl Sized + use<'a> {}
//~^ ERROR all possible in-scope parameters are already captured

struct Inherent;
impl Inherent {
    fn inherent(&self) -> impl Sized + use<'_> {}
    //~^ ERROR all possible in-scope parameters are already captured
}

trait Test<'a> {
    fn in_trait() -> impl Sized + use<'a, Self>;
    //~^ ERROR all possible in-scope parameters are already captured
}
impl<'a> Test<'a> for () {
    fn in_trait() -> impl Sized + use<'a> {}
    //~^ ERROR all possible in-scope parameters are already captured
}

fn main() {}
