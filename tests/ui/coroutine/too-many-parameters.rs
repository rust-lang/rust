#![feature(coroutines)]

fn main() {
    |(), ()| {
        //~^ error: too many parameters for a coroutine
        yield;
    };
}
