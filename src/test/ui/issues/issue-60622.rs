// ignore-tidy-linelength

#![deny(warnings)]

struct Borked {}

impl Borked {
    fn a(&self) {}
}

fn run_wild<T>(b: &Borked) {
    b.a::<'_, T>();
    //~^ ERROR cannot specify lifetime arguments explicitly if late bound lifetime parameters are present
    //~^^ ERROR wrong number of type arguments: expected 0, found 1
    //~^^^ WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn main() {}
