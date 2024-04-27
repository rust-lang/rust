#![feature(impl_trait_in_fn_trait_return)]
use std::fmt::Debug;

fn a<'a>() -> impl Fn(&'a u8) -> (impl Debug + '_) {
    |x| x
    //~^ ERROR expected generic lifetime parameter, found `'_`
}

fn _b<'a>() -> impl Fn(&'a u8) -> (impl Debug + 'a) {
    a()
}

fn main() {}
