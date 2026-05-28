//@revisions: edition2015 edition2024
//@[edition2015] edition:2015
//@[edition2024] edition:2024
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
