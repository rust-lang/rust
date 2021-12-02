// Checks that we report ABI mismatches for "const extern fn"
// compile-flags: -Z unleash-the-miri-inside-of-you

#![feature(const_extern_fn)]
#![allow(const_err)]

const extern "C" fn c_fn() {}

const fn call_rust_fn(my_fn: extern "Rust" fn()) {
    my_fn();
    //~^ ERROR could not evaluate static initializer
    //~| NOTE calling a function with calling convention C using calling convention Rust
    //~| NOTE inside `call_rust_fn`
}

static VAL: () = call_rust_fn(unsafe { std::mem::transmute(c_fn as extern "C" fn()) });
//~^ NOTE inside `VAL`

fn main() {}
