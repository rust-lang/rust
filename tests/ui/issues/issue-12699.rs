//@run
//@ignore-target-wasm32-unknown-unknown can't block the thread
//@ignore-target-sgx not supported
#![allow(deprecated)]

use std::thread;

fn main() {
    thread::sleep_ms(250);
}
