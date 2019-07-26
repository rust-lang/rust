// run-pass
#![allow(unused_must_use)]
// ignore-emscripten no threads support

use std::thread;

pub fn main() {
    let x = "Hello world!".to_string();
    thread::spawn(move|| {
        println!("{}", x);
    }).join();
}
