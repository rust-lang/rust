// compile-flags: -Zdrop-tracking

#![feature(generators, generator_trait)]

use std::ops::Generator;
use std::pin::Pin;

fn main() {
    let mut a = 5;
    let mut b = || {
        let d = 6;
        yield;
        _zzz(); // #break
        a = d;
    };
    Pin::new(&mut b).resume();
    //~^ ERROR this function takes 1 argument but 0 arguments were supplied
    // This type error is required to reproduce the ICE...
}

fn _zzz() {
    ()
}
