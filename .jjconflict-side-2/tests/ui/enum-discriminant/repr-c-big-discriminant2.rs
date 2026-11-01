//@ revisions: ptr32 ptr64
//@[ptr32] compile-flags: --target i686-unknown-linux-gnu
//@[ptr32] needs-llvm-components: x86
//@[ptr64] compile-flags: --target x86_64-unknown-linux-gnu
//@[ptr64] needs-llvm-components: x86
// GCC doesn't like cross-compilation
//@ ignore-backends: gcc
#![deny(repr_c_enums_larger_than_int)]

//@ add-minicore
#![feature(no_core)]
#![no_core]
extern crate minicore;
use minicore::*;

// Separate test since it suppresses other errors on ptr32:
// ensure we find the bad discriminant when it is implicitly computed by incrementing
// the previous discriminant.

#[repr(C)]
enum OverflowingEnum {
    NEG = -1,
    A = 2147483647, // i32::MAX
    B, // +1
    //[ptr32]~^ ERROR: enum discriminant overflowed
    //[ptr64]~^^ ERROR: discriminant does not fit into C `int`
    //[ptr64]~^^^ WARN: previously accepted
}

fn main() {}
