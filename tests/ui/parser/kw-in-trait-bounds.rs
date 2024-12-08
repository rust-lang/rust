//@ edition:2018

fn _f<F: fn(), G>(_: impl fn(), _: &dyn fn())
//~^ ERROR expected identifier, found keyword `fn`
//~| ERROR expected identifier, found keyword `fn`
//~| ERROR expected identifier, found keyword `fn`
//~| HELP use `Fn` to refer to the trait
//~| HELP use `Fn` to refer to the trait
//~| HELP use `Fn` to refer to the trait
where
G: fn(),
    //~^ ERROR expected identifier, found keyword `fn`
    //~| HELP use `Fn` to refer to the trait
{}

fn _g<A: struct, B>(_: impl struct, _: &dyn struct)
//~^ ERROR expected identifier, found keyword `struct`
//~| ERROR expected identifier, found keyword `struct`
//~| ERROR expected identifier, found keyword `struct`
//~| ERROR cannot find trait `r#struct` in this scope
//~| ERROR cannot find trait `r#struct` in this scope
//~| ERROR cannot find trait `r#struct` in this scope
//~| HELP  a trait with a similar name exists
//~| HELP  a trait with a similar name exists
//~| HELP  a trait with a similar name exists
//~| HELP  escape `struct` to use it as an identifier
//~| HELP  escape `struct` to use it as an identifier
//~| HELP  escape `struct` to use it as an identifier
where
    B: struct,
    //~^ ERROR expected identifier, found keyword `struct`
    //~| ERROR cannot find trait `r#struct` in this scope
    //~| HELP  a trait with a similar name exists
    //~| HELP  escape `struct` to use it as an identifier
{}

trait Struct {}

fn main() {}
