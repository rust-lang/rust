#![feature(min_const_generics)]

struct Bug<S: ?Sized> {
    A: [(); {
        let x: Option<Box<Self>> = None;
        //~^ ERROR generic `Self` types are currently not permitted in anonymous constants
        0
    }],
    B: S
}

fn main() {}
