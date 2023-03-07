#![feature(c_unwind, const_extern_fn)]

const extern "C" fn foo() {
    panic!() //~ ERROR evaluation of constant value failed
}

const _: () = foo();
// Ensure that the CTFE engine handles calls to `extern "C"` aborting gracefully

fn main() {
    let _ = foo();
}
