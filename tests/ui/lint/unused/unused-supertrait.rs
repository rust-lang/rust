#![deny(unused_must_use)]

fn it() -> impl ExactSizeIterator<Item = ()> {
    //~^ ERROR undefined opaque type
    let x: Box<dyn ExactSizeIterator<Item = ()>> = todo!();
    x
}

fn main() {
    it();
}
