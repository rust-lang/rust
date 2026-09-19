//@ add-minicore
//@ ignore-backends: gcc
//@ min-llvm-version: 22
//
//@ revisions: x86 x86_64 aarch64
//
//@ [x86] compile-flags: --target=i686-unknown-linux-gnu
//@ [x86] needs-llvm-components: x86
//@ [x86_64] compile-flags: --target=x86_64-unknown-linux-gnu
//@ [x86_64] needs-llvm-components: x86
//@ [aarch64] compile-flags: --target=aarch64-unknown-linux-gnu
//@ [aarch64] needs-llvm-components: aarch64
#![feature(explicit_tail_calls, rust_tail_cc, unsized_fn_params, no_core)]
#![allow(incomplete_features, internal_features)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

extern "C" {
    fn extract(_: [u8]) -> u8;
}

fn vanilla(b: [u8]) -> u8 {
    fn unsized_argument(x: [u8]) -> u8 {
        unsafe { extract(x) }
    }

    // Non-tail call.
    let _ = unsized_argument(b);

    become unsized_argument(b);
    //~^ ERROR unsized arguments cannot be used in a tail call
}

extern "tail" fn tailcc(b: &[u8]) -> u8 {
    // `extern "tail"` is special because we also can't unsized parameters in standard definitions
    // and calls.
    extern "tail" fn unsized_argument(x: [u8]) -> u8 {
        //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
        unsafe { extract(x) }
    }

    // Vanilla call.
    let _ = unsized_argument(*b);

    become unsized_argument(*b);
    //~^ ERROR unsized arguments cannot be used in a tail call
}
