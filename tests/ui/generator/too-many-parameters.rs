#![feature(generators)]

fn main() {
    |(), ()| {
        //~^ error: too many parameters for a generator
        yield;
    };
}
