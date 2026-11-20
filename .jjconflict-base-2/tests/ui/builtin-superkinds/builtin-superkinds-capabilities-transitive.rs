//@ run-pass
// Tests "transitivity" of super-builtin-kinds on traits. Here, if
// we have a Foo, we know we have a Bar, and if we have a Bar, we
// know we have a Send. So if we have a Foo we should know we have
// a Send. Basically this just makes sure rustc is using
// each_bound_trait_and_supertraits in type_contents correctly.


use std::sync::mpsc::{channel, Sender};

trait Bar : Send { }
trait Foo : Bar { }

impl <T: Send> Foo for T { }
impl <T: Send> Bar for T { }

fn foo<T: Foo + 'static>(val: T, chan: Sender<T>) {
    chan.send(val).unwrap();
}

pub fn main() {
    let (tx, rx) = channel();
    foo(31337, tx);
    assert_eq!(rx.recv().unwrap(), 31337);
}
