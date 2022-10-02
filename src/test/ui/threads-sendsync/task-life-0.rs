// run-pass
#![allow(unused_must_use)]
// ignore-emscripten no threads support
// pretty-expanded FIXME #23616

use std::thread;

pub fn main() {
    thread::spawn(move|| child("Hello".to_string()) ).join();
}

fn child(_s: String) {

}
