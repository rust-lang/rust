//@ run-pass
// Tests that the tail expr in null() has its type
// unified with the type *T, and so the type variable
// in that type gets resolved.


use std::mem;

fn null<T>() -> *const T {
    unsafe {
        mem::transmute(0_usize)
    }
}

pub fn main() { null::<isize>(); }
