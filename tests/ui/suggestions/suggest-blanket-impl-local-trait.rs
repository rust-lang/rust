// Ensure that the compiler include the blanklet implementation suggestion
// when inside a `impl` statement are used two local traits.
//
//@ edition:2021
use std::fmt;

trait LocalTraitOne { }

trait LocalTraitTwo { }

trait GenericTrait<T> {}

impl LocalTraitTwo for LocalTraitOne {}
//~^ ERROR expected a type, found a trait
//~| HELP you can add the `dyn` keyword if you want a trait object
//~| HELP alternatively use a blanket implementation to implement `LocalTraitTwo` for all types that also implement `LocalTraitOne`

impl fmt::Display for LocalTraitOne {
//~^ ERROR expected a type, found a trait
//~| HELP you can add the `dyn` keyword if you want a trait object
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
    }
}

impl fmt::Display for LocalTraitTwo + Send {
//~^ ERROR expected a type, found a trait
//~| HELP you can add the `dyn` keyword if you want a trait object
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!();
    }
}

impl LocalTraitOne for fmt::Display {}
//~^ ERROR expected a type, found a trait
//~| HELP you can add the `dyn` keyword if you want a trait object
//~| HELP alternatively use a blanket implementation to implement `LocalTraitOne` for all types that also implement `fmt::Display`


impl LocalTraitOne for fmt::Display + Send {}
//~^ ERROR expected a type, found a trait
//~| HELP you can add the `dyn` keyword if you want a trait object
//~| HELP alternatively use a blanket implementation to implement `LocalTraitOne` for all types that also implement `fmt::Display + Send`


impl<E> GenericTrait<E> for LocalTraitOne {}
//~^ ERROR expected a type, found a trait
//~| HELP you can add the `dyn` keyword if you want a trait object
//~| HELP alternatively use a blanket implementation to implement `GenericTrait<E>` for all types that also implement `LocalTraitOne`

trait GenericTraitTwo<T> {}

impl<T, E> GenericTraitTwo<E> for GenericTrait<T> {}
//~^ ERROR expected a type, found a trait
//~| HELP you can add the `dyn` keyword if you want a trait object
//~| HELP alternatively use a blanket implementation to implement `GenericTraitTwo<E>` for all types that also implement `GenericTrait<T>`

fn main() {}
