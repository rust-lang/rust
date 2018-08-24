#![feature(generators)]

fn main() { yield; }
//~^ ERROR yield statement outside
