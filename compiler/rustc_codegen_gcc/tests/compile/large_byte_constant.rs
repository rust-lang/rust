//@ memory-limit: 2G
// Compiler:

// An 8 MiB constant must compile within 2 GiB of address space. A backend that builds one rvalue
// per byte of a constant needs more than 6 GiB for it and runs out of memory.

#![crate_type = "lib"]

const SIZE: usize = 8 * 1024 * 1024;

// Not zero, so that the bytes cannot be emitted as a zero-initialized global.
pub static BYTES: [u8; SIZE] = [1; SIZE];
