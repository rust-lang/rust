#![feature(contracts)]
#![allow(incomplete_features)]

struct T;

impl T {
    #[core::contracts::ensures] //~ ERROR expected an `Fn(&_)` closure, found `()`
    fn b() {(loop)}
    //~^ ERROR expected `{`, found `)`
    //~| ERROR expected `{`, found `)`
}

fn main() {}
