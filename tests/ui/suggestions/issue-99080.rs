//@ aux-build:meow.rs

extern crate meow;

use meow::Meow;

fn needs_meow<T: Meow>(t: T) {}

fn main() {
    needs_meow(1usize);
    //~^ ERROR trait `Meow` is not implemented for `usize`
}

struct LocalMeow;

impl Meow for LocalMeow {}
