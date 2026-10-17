#![feature(coroutines)]

fn main() {
    yield || for i in 0 { }
    //~^ ERROR yield expression outside of coroutine literal
    //~| ERROR `{integer}` is not an iterator
    //~| ERROR `yield` can only be used in
}
