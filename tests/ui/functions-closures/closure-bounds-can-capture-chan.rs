// run-pass
// pretty-expanded FIXME #23616

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
