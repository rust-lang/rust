// run-pass
#![allow(unused_attributes)]
// aux-build:issue-29485.rs
// needs-unwind
// ignore-emscripten no threads

#[feature(recover)]

extern crate a;

fn main() {
    let _ = std::thread::spawn(move || {
        a::f(&mut a::X(0), g);
    }).join();
}

fn g() {
    panic!();
}
