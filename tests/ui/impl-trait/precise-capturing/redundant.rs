//@ compile-flags: -Zunstable-options --edition=2024
//@ check-pass

#![feature(precise_capturing)]
//~^ WARN the feature `precise_capturing` is incomplete

fn hello<'a>() -> impl use<'a> Sized {}
//~^ WARN all possible in-scope parameters are already captured

struct Inherent;
impl Inherent {
    fn inherent(&self) -> impl use<'_> Sized {}
    //~^ WARN all possible in-scope parameters are already captured
}

trait Test<'a> {
    fn in_trait() -> impl use<'a, Self> Sized;
    //~^ WARN all possible in-scope parameters are already captured
}
impl<'a> Test<'a> for () {
    fn in_trait() -> impl use<'a> Sized {}
    //~^ WARN all possible in-scope parameters are already captured
}

fn main() {}
