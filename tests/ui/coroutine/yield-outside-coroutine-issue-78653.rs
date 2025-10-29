#![feature(coroutines)]

fn main() {
    (|| for i in 0 { }).yield
    //~^ ERROR yield expression outside of coroutine literal
    //~| ERROR `{integer}` is not an iterator
    //~| ERROR `yield` can only be used in
}
