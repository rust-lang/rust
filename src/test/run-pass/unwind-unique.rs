// ignore-emscripten no threads support

#![feature(box_syntax)]

use std::thread;

fn f() {
    let _a: Box<_> = box 0;
    panic!();
}

pub fn main() {
    let t = thread::spawn(f);
    drop(t.join());
}
