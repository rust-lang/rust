//! Regression test for https://github.com/rust-lang/rust/issues/15094

#![feature(fn_traits, unboxed_closures)]

use std::{fmt, ops};

struct Debuger<T> {
    x: T
}

impl<T: fmt::Debug> ops::FnOnce<(),> for Debuger<T> {
    type Output = ();
    fn call_once(self, _args: ()) {
    //~^ ERROR `call_once` has an incompatible type for trait
    //~| NOTE expected signature `extern "rust-call" fn
    //~| NOTE found signature `fn
    //~| NOTE expected "rust-call" fn, found "Rust" fn
        println!("{:?}", self.x);
    }
}

fn make_shower<T>(x: T) -> Debuger<T> {
    Debuger { x: x }
}

pub fn main() {
    let show3 = make_shower(3);
    show3();
}
