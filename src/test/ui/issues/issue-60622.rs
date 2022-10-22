#![deny(warnings)]

struct Borked {}

impl Borked {
    fn a(&self) {}
}

fn run_wild<T>(b: &Borked) {
    b.a::<'_, T>();
    //~^ ERROR cannot specify lifetime arguments explicitly if late bound lifetime parameters are present
    //~| ERROR this associated function takes 0 generic arguments but 1 generic argument
}

fn main() {}
