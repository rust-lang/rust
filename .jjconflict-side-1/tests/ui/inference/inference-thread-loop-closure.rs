//! regression test for <https://github.com/rust-lang/rust/issues/20454>
//@ check-pass
#![allow(unused_must_use)]
use std::thread;

fn _foo() {
    thread::spawn(move || {
        // no need for -> ()
        loop {
            println!("hello");
        }
    })
    .join();
}

fn main() {}
