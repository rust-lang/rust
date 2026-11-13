// Regression test for https://github.com/rust-lang/rust/issues/158511.

#![allow(dead_code)]
#![deny(improper_ctypes)]

use std::num;

extern "C" {
    fn result_nonzero_u32_t(x: Result<num::NonZero<char>, ()>);
    //~^ ERROR `extern` block uses type `char`, which is not FFI-safe
}

fn main() {}
