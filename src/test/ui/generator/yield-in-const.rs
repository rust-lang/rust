#![feature(generators)]

const A: u8 = { yield 3u8; 3u8};
//~^ ERROR yield statement outside

fn main() {}
