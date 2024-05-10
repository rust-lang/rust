#![feature(impl_trait_in_fn_trait_return)]
use std::fmt::Debug;

fn a() -> impl Fn(&u8) -> impl Debug + '_ {
    //~^ ERROR ambiguous `+` in a type
    //~| ERROR cannot capture higher-ranked lifetime from outer `impl Trait`
    |x| x
}

fn b() -> impl Fn() -> impl Debug + Send {
    //~^ ERROR ambiguous `+` in a type
    || ()
}

fn main() {}
