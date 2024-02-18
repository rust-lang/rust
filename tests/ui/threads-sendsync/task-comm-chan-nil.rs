//@ run-pass

use std::sync::mpsc::channel;

// rustboot can't transmit nils across channels because they don't have
// any size, but rustc currently can because they do have size. Whether
// or not this is desirable I don't know, but here's a regression test.
pub fn main() {
    let (tx, rx) = channel();
    tx.send(()).unwrap();
    let n: () = rx.recv().unwrap();
    assert_eq!(n, ());
}
