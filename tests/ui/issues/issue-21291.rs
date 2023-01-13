// run-pass
// ignore-emscripten no threads support

// Regression test for unwrapping the result of `join`, issue #21291

use std::thread;

fn main() {
    thread::spawn(|| {}).join().unwrap()
}
