#![feature(fn_traits, unboxed_closures)]

use std::ops::FnMut;

struct S {
    x: isize,
    y: isize,
}

impl FnMut<(isize,)> for S {
    extern "rust-call" fn call_mut(&mut self, (z,): (isize,)) -> isize {
        self.x * self.y * z
    }
}

impl FnOnce<(isize,)> for S {
    type Output = isize;
    extern "rust-call" fn call_once(mut self, (z,): (isize,)) -> isize {
        self.call_mut((z,))
    }
}

fn main() {
    let mut s = S {
        x: 3,
        y: 3,
    };
    let ans = s("what");    //~ ERROR mismatched types
    let ans = s();
    //~^ ERROR this function takes 1 parameter but 0 parameters were supplied
    let ans = s("burma", "shave");
    //~^ ERROR this function takes 1 parameter but 2 parameters were supplied
}
