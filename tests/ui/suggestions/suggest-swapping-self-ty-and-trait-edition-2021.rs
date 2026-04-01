//@ edition:2021

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
//~| ERROR expected a type, found a trait

impl<'a, T> Enum<T> for Trait<'a, T> {}
//~^ ERROR expected trait, found enum `Enum`
//~| ERROR expected a type, found a trait

impl<'a, T> Union<T> for Trait<'a, T> {}
//~^ ERROR expected trait, found union `Union`
//~| ERROR expected a type, found a trait

fn main() {}
