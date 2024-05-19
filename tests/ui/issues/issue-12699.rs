//@ run-pass
//@ ignore-sgx not supported
#![allow(deprecated)]

use std::thread;

fn main() {
    thread::sleep_ms(250);
}
