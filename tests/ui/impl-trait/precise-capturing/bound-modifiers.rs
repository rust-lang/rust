//@ edition: 2021

fn polarity() -> impl Sized + ?use<> {}
//~^ ERROR expected identifier, found keyword `use`
//~| ERROR cannot find trait `r#use` in this scope

fn asyncness() -> impl Sized + async use<> {}
//~^ ERROR expected identifier, found keyword `use`
//~| ERROR cannot find trait `r#use` in this scope
//~| ERROR `async` trait bounds are unstable

fn constness() -> impl Sized + const use<> {}
//~^ ERROR expected identifier, found keyword `use`
//~| ERROR cannot find trait `r#use` in this scope
//~| ERROR const trait impls are experimental

fn binder() -> impl Sized + for<'a> use<> {}
//~^ ERROR expected identifier, found keyword `use`
//~| ERROR cannot find trait `r#use` in this scope

fn main() {}
