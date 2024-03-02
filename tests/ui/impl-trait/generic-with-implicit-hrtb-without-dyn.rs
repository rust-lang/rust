//@ revisions: edition2015 edition2021
//@[edition2021]edition:2021

#![allow(warnings)]

fn ice() -> impl AsRef<Fn(&())> {
    //[edition2015]~^ ERROR trait `AsRef<dyn for<'a> Fn(&'a ())>` is not implemented for `()`
    //[edition2021]~^^ ERROR: trait objects must include the `dyn` keyword [E0782]
    //[edition2021]~| ERROR trait `AsRef<dyn for<'a> Fn(&'a ())>` is not implemented for `()`
    todo!()
}

fn main() {}
