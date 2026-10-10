#![feature(postfix_match)]

use std::ops::Add;

//@ pretty-mode:expanded
//@ pp-exact:precedence.pp

macro_rules! repro {
    ($e:expr) => {
        $e.match {
            _ => {}
        }
    };
}

struct Struct {}

impl Add<Struct> for usize {
    type Output = ();
    fn add(self, _: Struct) -> () {
        ()
    }
}
pub fn main() {
    let a;

    repro!({ 1 } + 1);
    repro!(4 as usize);
    repro!(return);
    repro!(a = 42);
    repro!(|| {});
    repro!(42..101);
    repro!(1 + Struct {});
}
