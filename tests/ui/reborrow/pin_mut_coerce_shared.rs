use std::pin::Pin;

fn method(a: Pin<&()>) {}  //~NOTE function defined here

fn main() {
    let a = &mut ();
    let a = Pin::new(a);
    method(a);
    //~^ ERROR mismatched types
    //~| NOTE arguments to this function are incorrect
    //~| NOTE types differ in mutability
    //~| NOTE expected struct `Pin<&()>`
}
