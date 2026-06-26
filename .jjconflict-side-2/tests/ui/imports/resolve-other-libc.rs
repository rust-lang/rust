// Regression test for https://github.com/rust-lang/rust/issues/26043

//@ compile-flags: --extern libc=test.rlib

// The error shall NOT be something similar to the following, because it
// indicates that `libc` was wrongly resolved to `libc` shipped with the
// compiler:
//
//   error[E0658]: use of unstable library feature `rustc_private`: \
//           this crate is being loaded from the sysroot
//
extern crate libc; //~ ERROR: extern location for libc does not exist: test.rlib

fn main() {}
