#![feature(contracts)]
#![allow(incomplete_features)]

struct T;

impl T {
    #[core::contracts::ensures] //~ ERROR `ensures` attribute requires an argument
    fn b() {(loop)}
    //~^ ERROR expected `{`, found `)`
    //~| ERROR expected `{`, found `)`
}

fn main() {}
