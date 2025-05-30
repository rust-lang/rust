// Checks that we report ABI mismatches for "const extern fn"
//@ compile-flags: -Z unleash-the-miri-inside-of-you

const extern "C" fn c_fn() {}

const fn call_rust_fn(my_fn: extern "Rust" fn()) {
    my_fn(); //~ NOTE inside `call_rust_fn`
    //~^ NOTE the failure occurred here
}

static VAL: () = call_rust_fn(unsafe { std::mem::transmute(c_fn as extern "C" fn()) });
//~^ NOTE evaluation of static initializer failed here
//~| ERROR calling a function with calling convention "C" using calling convention "Rust"

fn main() {}

//~? WARN skipping const checks
