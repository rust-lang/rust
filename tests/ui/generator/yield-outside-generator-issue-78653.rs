#![feature(generators)]

fn main() {
    yield || for i in 0 { }
    //~^ ERROR yield expression outside of generator literal
    //~| ERROR `{integer}` is not an iterator
}
