#![feature(fn_traits, unboxed_closures)]

use std::{fmt, ops};

struct Debugger<T> {
    x: T
}

impl<T: fmt::Debug> ops::FnOnce<(),> for Debugger<T> {
    type Output = ();
    fn call_once(self, _args: ()) {
    //~^ ERROR `call_once` has an incompatible type for trait
    //~| expected signature `extern "rust-call" fn
    //~| found signature `fn
        println!("{:?}", self.x);
    }
}

fn make_shower<T>(x: T) -> Debugger<T> {
    Debugger { x: x }
}

pub fn main() {
    let show3 = make_shower(3);
    show3();
}
