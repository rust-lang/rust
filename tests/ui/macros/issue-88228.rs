//@ compile-flags: -Z deduplicate-diagnostics=yes
//@ edition:2018

mod hey { //~ HELP consider importing this derive macro
    //~^ HELP consider importing this macro
    pub use Copy as Bla;
    pub use std::println as bla;
}

#[derive(Bla)]
//~^ ERROR cannot find derive macro `Bla`
//~| NOTE `Bla` is a trait, not a derive macro
//~| HELP consider implementing `Bla` for your type manually
struct A;

#[derive(println)]
//~^ ERROR cannot find derive macro `println`
//~| NOTE `println` is in scope, but it is a function-like macro
struct B;

fn main() {
    bla!();
    //~^ ERROR cannot find macro `bla`
}
