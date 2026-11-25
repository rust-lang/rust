//@ compile-flags: -Znext-solver

#![feature(unsafe_binders)]
#![expect(incomplete_features)]
#![deny(improper_ctypes)]

extern "C" {
    fn exit_1(x: unsafe<'a> &'a u32);
    fn exit_2(x: unsafe<'a> &'a ());
    //~^ ERROR `extern` block uses type `()`, which is not FFI-safe
}

fn main() {}
