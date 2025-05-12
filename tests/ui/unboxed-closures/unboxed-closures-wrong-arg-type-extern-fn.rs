// Tests that unsafe extern fn pointers do not implement any Fn traits.

use std::ops::{Fn, FnMut, FnOnce};

unsafe fn square(x: isize) -> isize {
    x * x
}
// note: argument type here is `isize`, not `&isize`

fn call_it<F: Fn(&isize) -> isize>(_: &F, _: isize) -> isize {
    0
}
fn call_it_mut<F: FnMut(&isize) -> isize>(_: &mut F, _: isize) -> isize {
    0
}
fn call_it_once<F: FnOnce(&isize) -> isize>(_: F, _: isize) -> isize {
    0
}

fn a() {
    let x = call_it(&square, 22);
    //~^ ERROR E0277
}

fn b() {
    let y = call_it_mut(&mut square, 22);
    //~^ ERROR E0277
}

fn c() {
    let z = call_it_once(square, 22);
    //~^ ERROR E0277
}

fn main() {}
