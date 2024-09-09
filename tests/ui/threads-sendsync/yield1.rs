//@ run-pass

#![allow(unused_must_use)]
#![allow(unused_mut)]
//@ needs-threads

use std::thread;

pub fn main() {
    let mut result = thread::spawn(child);
    println!("1");
    thread::yield_now();
    result.join();
}

fn child() {
    println!("2");
}
