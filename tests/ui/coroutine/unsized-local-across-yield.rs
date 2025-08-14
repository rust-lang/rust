#![feature(coroutine_trait)]
#![feature(coroutines)]
#![feature(unsized_locals)]
//~^ WARN the feature `unsized_locals` is incomplete and may not be safe to use and/or cause compiler crashes

use std::ops::Coroutine;

fn across() -> impl Coroutine {
    #[coroutine]
    move || {
        let b: [u8] = *(Box::new([]) as Box<[u8]>);
        //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time

        yield;

        for elem in b.iter() {}
    }
}

fn main() {
    across();
}
