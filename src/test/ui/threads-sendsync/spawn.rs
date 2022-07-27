// run-pass
// ignore-emscripten no threads support
// ignore-uefi no threads support

use std::thread;

pub fn main() {
    thread::spawn(move|| child(10)).join().ok().unwrap();
}

fn child(i: isize) { println!("{}", i); assert_eq!(i, 10); }
