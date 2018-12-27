#![allow(unused_must_use)]
#![allow(unused_mut)]
// ignore-emscripten no threads support

use std::thread;

pub fn main() {
    let mut result = thread::spawn(child);
    println!("1");
    thread::yield_now();
    result.join();
}

fn child() { println!("2"); }
