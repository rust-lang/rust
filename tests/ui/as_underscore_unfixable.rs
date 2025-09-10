//@no-rustfix

#![warn(clippy::as_underscore)]

fn main() {
    // From issue #15282
    let f = async || ();
    let _: Box<dyn FnOnce() -> _> = Box::new(f) as _;
    //~^ as_underscore

    let barr = || (|| ());
    let _: Box<dyn Fn() -> _> = Box::new(barr) as _;
    //~^ as_underscore
}
