//@ compile-flags: -Znext-solver

#![feature(unsafe_binders)]
#![expect(incomplete_features)]
#![deny(improper_ctypes)]

extern "C" {
    fn exit_1(x: unsafe<'a> &'a u32);
    //~^ ERROR `extern` block uses type `unsafe<'a> &'a u32`, which is not FFI-safe
    fn exit_2(x: unsafe<'a> &'a ());
    //~^ ERROR `extern` block uses type `unsafe<'a> &'a ()`, which is not FFI-safe
}

fn main() {}
