// run-pass
#![allow(unused_must_use)]
// ignore-emscripten no threads support

use std::thread;

pub fn main() { test00(); }

fn start() { println!("Started / Finished task."); }

fn test00() {
    thread::spawn(move|| start() ).join();
    println!("Completing.");
}
