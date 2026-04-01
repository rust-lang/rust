#![feature(coroutines)]

fn main() { yield; }
//~^ ERROR yield expression outside
//~| ERROR `yield` can only be used in
