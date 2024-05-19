//@ run-pass
//@ needs-threads

// Regression test for unwrapping the result of `join`, issue #21291

use std::thread;

fn main() {
    thread::spawn(|| {}).join().unwrap()
}
