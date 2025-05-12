#![feature(coroutines)]

static B: u8 = { yield 3u8; 3u8};
//~^ ERROR yield expression outside
//~| ERROR `yield` can only be used in

fn main() {}
