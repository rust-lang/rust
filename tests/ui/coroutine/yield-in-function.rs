#![feature(coroutines)]

fn main() { yield; }
//~^ ERROR yield expression outside
