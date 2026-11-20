//@ run-pass
#![allow(unused_must_use)]
//@ needs-threads

use std::thread;

pub fn main() {
    thread::spawn(move || child("Hello".to_string())).join();
}

fn child(_s: String) {}
