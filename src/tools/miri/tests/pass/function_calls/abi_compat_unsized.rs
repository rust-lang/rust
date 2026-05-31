#![feature(unsized_fn_params)]

use std::mem;

#[repr(transparent)]
#[derive(Copy, Clone)]
struct Wrapper<T: ?Sized>(T);

fn unsized_arg(_x: [i32]) {}

fn main() {
    let ptr = unsized_arg as fn([i32]);
    let ptr: fn(Wrapper<[i32]>) = unsafe { mem::transmute(ptr) };
    let w: Box<Wrapper<[i32]>> = Box::new(Wrapper([1, 2]));
    ptr(*w);
}
