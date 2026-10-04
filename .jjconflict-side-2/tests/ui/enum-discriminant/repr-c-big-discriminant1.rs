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

#[repr(C)]
enum OverflowingEnum1 {
    A = 9223372036854775807, // i64::MAX
    //[ptr32]~^ ERROR: literal out of range
    //[ptr64]~^^ ERROR: discriminant does not fit into C `int` nor into C `unsigned int`
    //[ptr64]~^^^ WARN: previously accepted
}

#[repr(C)]
enum OverflowingEnum2 {
    A = -2147483649, // i32::MIN-1
    //[ptr32]~^ ERROR: literal out of range
    //[ptr64]~^^ ERROR: discriminant does not fit into C `int` nor into C `unsigned int`
    //[ptr64]~^^^ WARN: previously accepted
}

#[repr(C)]
enum OverflowingEnum3a {
    A = 2147483648, // i32::MAX+1
    //[ptr32]~^ ERROR: literal out of range
    B = -1,
    //[ptr64]~^ ERROR: discriminant does not fit into C `unsigned int`, and a previous
    //[ptr64]~^^ WARN: previously accepted
}
#[repr(C)]
enum OverflowingEnum3b {
    B = -1,
    A = 2147483648, // i32::MAX+1
    //[ptr32]~^ ERROR: literal out of range
    //[ptr64]~^^ ERROR: discriminant does not fit into C `int`, and a previous
    //[ptr64]~^^^ WARN: previously accepted
}

const I64_MAX: i64 = 9223372036854775807;

#[repr(C)]
enum OverflowingEnum4 {
    A = I64_MAX as isize,
    //[ptr64]~^ ERROR: discriminant does not fit into C `int` nor into C `unsigned int`
    //[ptr64]~^^ WARN: previously accepted
    // No warning/error on 32bit targets, but the `as isize` hints that wrapping is occurring.
}

// Enums like this are found in the ecosystem, let's make sure they get accepted.
#[repr(C)]
#[allow(overflowing_literals)]
enum OkayEnum {
    A = 0,
    O = 0xffffffff,
}

fn main() {}
