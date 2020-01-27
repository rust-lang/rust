// Checks that we report ABI mismatches for "const extern fn"
// compile-flags: -Z unleash-the-miri-inside-of-you

#![feature(const_extern_fn)]

const extern "C" fn c_fn() {}

const fn call_rust_fn(my_fn: extern "Rust" fn()) {
    my_fn(); //~ ERROR any use of this value will cause an error
    //~^ WARN skipping const checks
}

const VAL: () = call_rust_fn(unsafe { std::mem::transmute(c_fn as extern "C" fn()) });
//~^ WARN skipping const checks

fn main() {}
