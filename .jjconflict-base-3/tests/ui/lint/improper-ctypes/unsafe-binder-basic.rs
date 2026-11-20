#![feature(unsafe_binders)]
#![expect(incomplete_features)]
#![deny(improper_ctypes)]

extern "C" {
    fn exit_2(x: unsafe<'a> &'a ());
    //~^ ERROR `extern` block uses type `unsafe<'a> &'a ()`, which is not FFI-safe
}

fn main() {}
