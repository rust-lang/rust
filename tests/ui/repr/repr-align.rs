//@ revisions: win notwin
//@ add-minicore
//@ [win] compile-flags: --target x86_64-pc-windows-msvc
//@ [win] needs-llvm-components: x86
//@ [notwin] compile-flags: --target x86_64-unknown-linux-gnu
//@ [notwin] needs-llvm-components: x86

#![feature(no_core, lang_items)]
#![no_core]
#![allow(dead_code)]
extern crate minicore;
use minicore::*;

#[repr(align(16.0))] //[notwin,win]~ ERROR: invalid `repr(align)` attribute: not an unsuffixed integer
struct S0(i32);

#[repr(align(15))] //[notwin,win]~ ERROR: invalid `repr(align)` attribute: not a power of two
struct S1(i32);

#[repr(align(4294967296))] //[notwin,win]~ ERROR: invalid `repr(align)` attribute: larger than 2^29
struct S2(i32);

#[repr(align(536870912))] //[win]~ ERROR: alignment must not be greater than 8192 bytes for COFF targets
                          // notwin: this is the largest accepted alignment
struct S3(i32);

#[repr(align(0))] //[notwin,win]~ ERROR: invalid `repr(align)` attribute: not a power of two
struct S4(i32);

#[repr(align(16384))]  //[win]~ ERROR: alignment must not be greater than 8192 bytes for COFF targets
struct S5(i32);

#[repr(align(16.0))] //[notwin,win]~ ERROR: invalid `repr(align)` attribute: not an unsuffixed integer
enum E0 { A, B }

#[repr(align(15))] //[notwin,win]~ ERROR: invalid `repr(align)` attribute: not a power of two
enum E1 { A, B }

#[repr(align(4294967296))] //[notwin,win]~ ERROR: invalid `repr(align)` attribute: larger than 2^29
enum E2 { A, B }

#[repr(align(536870912))] //[win]~ ERROR: alignment must not be greater than 8192 bytes for COFF targets
                          // notwin: this is the largest accepted alignment
enum E3 { A, B }

#[repr(align(0))] //[notwin,win]~ ERROR: invalid `repr(align)` attribute: not a power of two
enum E4 { A, B }

#[repr(align(16384))]  //[win]~ ERROR: alignment must not be greater than 8192 bytes for COFF targets
enum E5 { A, B }

fn main() {}
