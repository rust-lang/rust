// Ensure that the compiler include the blanklet implementation suggestion
// when inside a `impl` statment are used two local traits.
//
// edition:2021
use std::fmt;

trait LocalTraitOne { }

trait LocalTraitTwo { }

trait GenericTrait<T> {}

impl LocalTraitTwo for LocalTraitOne {}
//~^ ERROR trait objects must include the `dyn` keyword

impl fmt::Display for LocalTraitOne {
//~^ ERROR trait objects must include the `dyn` keyword
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
    }
}

impl fmt::Display for LocalTraitTwo + Send {
//~^ ERROR trait objects must include the `dyn` keyword
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
    }
}

impl LocalTraitOne for fmt::Display {}
//~^ ERROR trait objects must include the `dyn` keyword

impl LocalTraitOne for fmt::Display + Send {}
//~^ ERROR trait objects must include the `dyn` keyword

impl<E> GenericTrait<E> for LocalTraitOne {}
//~^ ERROR trait objects must include the `dyn` keyword

trait GenericTraitTwo<T> {}

impl<T, E> GenericTraitTwo<E> for GenericTrait<T> {}
//~^ ERROR trait objects must include the `dyn` keyword

fn main() {}
