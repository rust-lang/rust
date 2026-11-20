#![warn(clippy::as_pointer_underscore)]
#![crate_type = "lib"]
#![no_std]

struct S;

fn f(s: &S) -> usize {
    &s as *const _ as usize
    //~^ as_pointer_underscore
}

fn g(s: &mut S) -> usize {
    s as *mut _ as usize
    //~^ as_pointer_underscore
}
