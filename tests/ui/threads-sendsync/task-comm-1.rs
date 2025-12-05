//@ run-pass
#![allow(unused_must_use)]
//@ needs-threads

use std::thread;

pub fn main() {
    test00();
}

fn start() {
    println!("Started / Finished task.");
}

fn test00() {
    thread::spawn(move || start()).join();
    println!("Completing.");
}
