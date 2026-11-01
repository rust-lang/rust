//@ run-pass

use std::sync::mpsc::channel;

fn foo<F:FnOnce()+Send>(blk: F) {
    blk();
}

pub fn main() {
    let (tx, rx) = channel();
    foo(move || {
        tx.send(()).unwrap();
    });
    rx.recv().unwrap();
}
