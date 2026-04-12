//@ run-pass
//@ check-stdout
//@ check-run-results

#![feature(const_destruct)]
use std::marker::Destruct;
struct A {
    _a: String,
}

impl Destruct for A {
    unsafe fn drop_in_place(_to_drop: *mut Self) {
        println!("Hey i was dropped");
    }
}

fn main() {
    let _a = A { _a: String::new() };
}
