#![feature(generators)]

fn main() { yield; }
//~^ ERROR yield expression outside
