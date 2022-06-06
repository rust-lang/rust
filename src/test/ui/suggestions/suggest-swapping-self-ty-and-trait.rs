pub trait Trait<'a, T> {}

pub struct Struct<T> {}

impl<'a, T> Struct<T> for Trait<'a, T> {}
//~^ ERROR expected trait, found struct `Struct`
//~| WARNING trait objects without an explicit `dyn` are deprecated
//~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!

fn main() {}
