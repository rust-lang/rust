#![feature(impl_trait_in_fn_trait_return)]
use std::fmt::Debug;

fn a() -> impl Fn(&u8) -> (impl Debug + '_) {
    //~^ ERROR higher kinded lifetime bounds on nested opaque types are not supported yet
    |x| x
}

fn b() -> impl for<'a> Fn(&'a u8) -> (impl Debug + 'a) {
    //~^ ERROR higher kinded lifetime bounds on nested opaque types are not supported yet
    |x| x
}

fn c() -> impl for<'a> Fn(&'a u8) -> (impl Debug + '_) {
    //~^ ERROR higher kinded lifetime bounds on nested opaque types are not supported yet
    |x| x
}

fn d() -> impl Fn() -> (impl Debug + '_) {
    //~^ ERROR missing lifetime specifier
    || ()
}

fn main() {}
