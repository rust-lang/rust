// run-pass
#![allow(unused_must_use)]
// ignore-emscripten no threads support

use std::thread;

macro_rules! expr { ($e: expr) => { $e } }

macro_rules! spawn {
    ($($code: tt)*) => {
        expr!(thread::spawn(move|| {$($code)*}).join())
    }
}

pub fn main() {
    spawn! {
        println!("stmt");
    };
    let _ = spawn! {
        println!("expr");
    };
}
