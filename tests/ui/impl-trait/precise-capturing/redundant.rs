//@ compile-flags: -Zunstable-options --edition=2024
//@ revisions: normal rpitit
//@[normal] check-pass

#![feature(precise_capturing)]

fn hello<'a>() -> impl Sized + use<'a> {}
//[normal]~^ WARN all possible in-scope parameters are already captured

struct Inherent;
impl Inherent {
    fn inherent(&self) -> impl Sized + use<'_> {}
    //[normal]~^ WARN all possible in-scope parameters are already captured
}

#[cfg(rpitit)]
trait Test<'a> {
    fn in_trait() -> impl Sized + use<'a, Self>;
    //[rpitit]~^ ERROR `use<...>` precise capturing syntax is currently not allowed in return-position `impl Trait` in traits
}
#[cfg(rpitit)]
impl<'a> Test<'a> for () {
    fn in_trait() -> impl Sized + use<'a> {}
    //[rpitit]~^ ERROR `use<...>` precise capturing syntax is currently not allowed in return-position `impl Trait` in traits
}

fn main() {}
