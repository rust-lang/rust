#![feature(coroutines)]

fn main() {
    #[coroutine]
    |(), ()| {
        //~^ error: too many parameters for a coroutine
        yield;
    };
}
