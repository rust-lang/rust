//! Regression test for unwrapping the result of `join`, issue https://github.com/rust-lang/rust/issues/21291
//@ run-pass
//@ needs-threads

use std::thread;

fn main() {
    thread::spawn(|| {}).join().unwrap()
}
