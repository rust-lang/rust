//@ aux-build:meow.rs

extern crate meow;

use meow::Meow;

fn needs_meow<T: Meow>(t: T) {}

fn main() {
    needs_meow(1usize);
    //~^ ERROR the trait bound `usize: Meow` is not satisfied
}

struct LocalMeow;

impl Meow for LocalMeow {}
