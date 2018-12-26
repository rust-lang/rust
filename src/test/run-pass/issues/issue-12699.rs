// run-pass
// ignore-wasm32-bare can't block the thread
#![allow(deprecated)]

use std::thread;

fn main() {
    thread::sleep_ms(250);
}
