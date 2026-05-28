//@ run-rustfix

#[allow(unused)]
struct Struct<T>(T);

impl<T> Struct<T> where T:: std::fmt::Display {
//~^ ERROR expected `:` followed by trait or lifetime
//~| HELP use single colon
}

fn main() {}
