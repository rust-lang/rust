#![feature(explicit_tail_calls)]
#![allow(incomplete_features)]

fn main() {
    f(0);
}

fn f(x: u32) {
    let g = unsafe { std::mem::transmute::<fn(i32), fn(u32)>(g) };
    become g(x); // FIXME ideally this should also be involved in the error somehow,
    // but by the time we pass the argument, `f`'s stackframe has already been popped.
}

fn g(_: i32) {}
//~^ error: type i32 passing argument of type u32
