//! Regression test for https://github.com/rust-lang/rust/issues/12041

use std::sync::mpsc::channel;
use std::thread;

fn main() {
    let (tx, rx) = channel();
    let _t = thread::spawn(move|| -> () {
        loop {
            let tx = tx;
            //~^ ERROR: use of moved value: `tx`
            tx.send(1);
        }
    });
}
