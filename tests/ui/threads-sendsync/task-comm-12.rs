// run-pass
#![allow(unused_must_use)]
#![allow(unused_mut)]
// ignore-emscripten no threads support

use std::thread;

pub fn main() { test00(); }

fn start(_task_number: isize) { println!("Started / Finished task."); }

fn test00() {
    let i: isize = 0;
    let mut result = thread::spawn(move|| {
        start(i)
    });

    // Sleep long enough for the thread to finish.
    let mut i = 0_usize;
    while i < 10000 {
        thread::yield_now();
        i += 1;
    }

    // Try joining threads that have already finished.
    result.join();

    println!("Joined task.");
}
