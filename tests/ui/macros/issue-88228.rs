// compile-flags: -Z deduplicate-diagnostics=yes
// edition:2018

mod hey {
    pub use Copy as Bla;
    pub use std::println as bla;
}

#[derive(Bla)]
//~^ ERROR cannot find derive macro `Bla`
//~| NOTE consider importing this derive macro
struct A;

#[derive(println)]
//~^ ERROR cannot find derive macro `println`
//~|`println` is in scope, but it is a function-like macro
struct B;

fn main() {
    bla!();
    //~^ ERROR cannot find macro `bla`
    //~| NOTE consider importing this macro
}
