#![feature(fn_traits, unboxed_closures)]

use std::mem::transmute;

#[repr(align(4))]
struct Zst;

fn foo<F: FnOnce()>(_: F) {
    // Calls the given F FnOnce, but passing an over-aligned ZST instead of the closure / function item
    let f = unsafe { transmute::<extern "rust-call" fn(F, ()), fn(Zst)>(F::call_once) };
    f(Zst)
}

fn main() {
    foo(move || {
        //~^ERROR: /calling a function whose parameter #1 has type .* passing argument of type Zst/
        println!("non-capturing closure");
    });
}
