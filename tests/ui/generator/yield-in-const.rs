#![feature(generators)]

const A: u8 = { yield 3u8; 3u8};
//~^ ERROR yield expression outside

fn main() {}
