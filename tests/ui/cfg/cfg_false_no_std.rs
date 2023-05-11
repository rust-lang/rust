// Currently no error because the panic handler is supplied by libstd linked though the empty
// library, but the desirable behavior is unclear (see comments in cfg_false_lib.rs).

// check-pass
// aux-build: cfg_false_lib.rs

#![no_std]

extern crate cfg_false_lib as _;

fn main() {}
