//@ compile-flags: -Zunstable-options --edition=2024
//@ check-pass

#![feature(precise_capturing_in_traits)]

fn hello<'a>() -> impl Sized + use<'a> {}
//~^ WARN all possible in-scope parameters are already captured

struct Inherent;
impl Inherent {
    fn inherent(&self) -> impl Sized + use<'_> {}
    //~^ WARN all possible in-scope parameters are already captured
}

trait Test<'a> {
    fn in_trait() -> impl Sized + use<'a, Self>;
    //~^ WARN all possible in-scope parameters are already captured
}
impl<'a> Test<'a> for () {
    fn in_trait() -> impl Sized + use<'a> {}
    //~^ WARN all possible in-scope parameters are already captured
}

fn main() {}
