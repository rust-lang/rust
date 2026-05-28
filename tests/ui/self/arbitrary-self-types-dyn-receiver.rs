//@ run-pass
#![feature(arbitrary_self_types)]

use std::ops::Receiver;

trait Trait {
    fn foo(self: &dyn Receiver<Target=Self>);
}

struct Thing;
impl Trait for Thing {
    fn foo(self: &dyn Receiver<Target=Self>) {
        println!("huh???");
    }
}

fn main() {
    let x = Box::new(Thing);
    let y: &dyn Receiver<Target=Thing> = &x;
    y.foo();
}
