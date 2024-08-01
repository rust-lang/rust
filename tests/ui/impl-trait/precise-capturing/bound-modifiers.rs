//@ edition: 2021

#![feature(precise_capturing)]

fn polarity() -> impl Sized + ?use<> {}
//~^ ERROR expected identifier, found keyword `use`
//~| ERROR cannot find trait `r#use` in this scope
//~| WARN relaxing a default bound only does something for `?Sized`
//~| WARN relaxing a default bound only does something for `?Sized`

fn asyncness() -> impl Sized + async use<> {}
//~^ ERROR expected identifier, found keyword `use`
//~| ERROR cannot find trait `r#use` in this scope
//~| ERROR async closures are unstable

fn constness() -> impl Sized + const use<> {}
//~^ ERROR expected identifier, found keyword `use`
//~| ERROR cannot find trait `r#use` in this scope
//~| ERROR const trait impls are experimental

fn binder() -> impl Sized + for<'a> use<> {}
//~^ ERROR expected identifier, found keyword `use`
//~| ERROR cannot find trait `r#use` in this scope

fn main() {}
