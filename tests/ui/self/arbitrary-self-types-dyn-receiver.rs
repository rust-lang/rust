//@ check-fail
#![feature(arbitrary_self_types)]

use std::ops::Receiver;

trait Trait {
    fn foo(self: &dyn Receiver<Target=Self>);
    //~^ ERROR: the trait `std::ops::Receiver` is not dyn compatible
    //~| ERROR: the trait `std::ops::Receiver` is not dyn compatible
}

struct Thing;
impl Trait for Thing {
    fn foo(self: &dyn Receiver<Target=Self>) {
        //~^ ERROR: the trait `std::ops::Receiver` is not dyn compatible
        //~| ERROR: the trait `std::ops::Receiver` is not dyn compatible
        //~| ERROR: the trait `std::ops::Receiver` is not dyn compatible
        println!("huh???");
    }
}

fn main() {
    let x = Box::new(Thing);
    let y: &dyn Receiver<Target=Thing> = &x;
    //~^ ERROR: the trait `std::ops::Receiver` is not dyn compatible
    y.foo();
}
