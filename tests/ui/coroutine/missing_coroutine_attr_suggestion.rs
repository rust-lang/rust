//@ run-rustfix

#![feature(coroutines)]

fn main() {
    let _ = || yield;
    //~^ ERROR `yield` can only be used
}
