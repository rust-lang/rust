// edition:2021

pub trait Trait<'a, T> {}

pub struct Struct<T> {}

impl<'a, T> Struct<T> for Trait<'a, T> {}
//~^ ERROR expected trait, found struct `Struct`
//~| ERROR trait objects must include the `dyn` keyword

fn main() {}
