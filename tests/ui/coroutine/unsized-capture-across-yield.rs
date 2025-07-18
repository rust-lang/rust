#![feature(coroutine_trait)]
#![feature(coroutines)]

use std::ops::Coroutine;

fn capture() -> impl Coroutine {
    let b: [u8] = *(Box::new([]) as Box<[u8]>); //~ERROR he size for values of type `[u8]` cannot be known at compilation time
    #[coroutine]
    move || {
        println!("{:?}", &b);

        yield;

        for elem in b.iter() {}
    }
}

fn main() {
    capture();
}
