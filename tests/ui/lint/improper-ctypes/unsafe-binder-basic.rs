//@ check-pass
#![feature(unsafe_binders)]
#![expect(incomplete_features)]
#![deny(improper_ctypes)]


extern "C" {
    fn exit_2(x: unsafe<'a> &'a ());
}

fn main() {}
