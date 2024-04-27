pub trait Trait<'a, T> {}

pub struct Struct<T>;
//~^ ERROR `T` is never used
pub enum Enum<T> {}
//~^ ERROR `T` is never used

pub union Union<T> {
    //~^ ERROR `T` is never used
    f1: usize,
}

impl<'a, T> Struct<T> for Trait<'a, T> {}
//~^ ERROR expected trait, found struct `Struct`
//~| WARNING trait objects without an explicit `dyn` are deprecated
//~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!

impl<'a, T> Enum<T> for Trait<'a, T> {}
//~^ ERROR expected trait, found enum `Enum`
//~| WARNING trait objects without an explicit `dyn` are deprecated
//~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!

impl<'a, T> Union<T> for Trait<'a, T> {}
//~^ ERROR expected trait, found union `Union`
//~| WARNING trait objects without an explicit `dyn` are deprecated
//~| WARNING this is accepted in the current edition (Rust 2015) but is a hard error in Rust 2021!

fn main() {}
