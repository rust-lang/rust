use std::fmt::Debug;

fn in_adt_in_return() -> Vec<impl Debug> { panic!() }
//~^ ERROR type annotations needed

fn main() {}
