//@ run-pass
// Tests the ability for the Self type in default methods to use
// capabilities granted by builtin kinds as supertraits.


use std::sync::mpsc::{Sender, channel};

trait Foo : Send + Sized + 'static {
    fn foo(self, tx: Sender<Self>) {
        tx.send(self).unwrap();
    }
}

impl <T: Send + 'static> Foo for T { }

pub fn main() {
    let (tx, rx) = channel();
    1193182.foo(tx);
    assert_eq!(rx.recv().unwrap(), 1193182);
}
