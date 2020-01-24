#![feature(generators)]

fn main() {
    |(), ()| {  //~ error: too many parameters for generator
        yield;
    };
}
