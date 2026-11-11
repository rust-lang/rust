//@ run-pass
#![allow(unused_must_use)]
//@ needs-threads

// Issue #922

// This test is specifically about spawning temporary closures.

use std::thread;

fn f() {}

pub fn main() {
    thread::spawn(move || f()).join();
}
