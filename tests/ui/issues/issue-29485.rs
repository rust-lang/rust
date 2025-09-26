//@ run-pass
#![allow(unused_attributes)]
//@ aux-build:issue-29485.rs
//@ needs-unwind
//@ needs-threads
//@ ignore-backends: gcc

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
