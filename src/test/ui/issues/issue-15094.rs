#![feature(fn_traits, unboxed_closures)]

use std::{fmt, ops};

struct Debuger<T> {
    x: T
}

impl<T: fmt::Debug> ops::FnOnce<(),> for Debuger<T> {
    type Output = ();
    fn call_once(self, _args: ()) {
    //~^ ERROR `call_once` has an incompatible type for trait
    //~| expected type `extern "rust-call" fn
    //~| found type `fn
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
