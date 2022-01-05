// run-fail
// error-pattern: panicked while processing panic
#![allow(stable_features)]

// ignore-emscripten no threads support

#![feature(std_panic)]
#![feature(panic_update_hook)]

use std::panic;

fn main() {
    panic::update_hook(|_prev| {
        panic!("inside update_hook");
    })
}
