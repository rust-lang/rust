//@ run-pass
//@ ignore-sgx not supported
//@ ignore-emscripten
// FIXME: test hangs on emscripten
#![allow(deprecated)]
#![allow(unused_imports)]

use std::thread;

fn main() {
    thread::sleep_ms(250);
}
