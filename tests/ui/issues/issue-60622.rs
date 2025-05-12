#![deny(warnings)]

struct Borked {}

impl Borked {
    fn a(&self) {}
}

fn run_wild<T>(b: &Borked) {
    b.a::<'_, T>();
    //~^ ERROR cannot specify lifetime arguments explicitly if late bound lifetime parameters are present
    //~| ERROR method takes 0 generic arguments but 1 generic argument
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn main() {}
