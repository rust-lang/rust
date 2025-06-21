//@ revisions: win notwin
//@[win] only-windows
//@[notwin] ignore-windows

#![feature(fn_align)]
#![allow(dead_code)]

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

#[align(16.0)] //[notwin,win]~ ERROR: invalid alignment value: not an unsuffixed integer
fn f0() {}

#[align(15)] //[notwin,win]~ ERROR: invalid alignment value: not a power of two
fn f1() {}

#[align(4294967296)] //[notwin,win]~ ERROR: alignment value: larger than 2^29
fn f2() {}

#[align(536870912)] //[win]~ ERROR: alignment must not be greater than 8192 bytes for COFF targets
                          // notwin: this is the largest accepted alignment
fn f3() {}

#[align(0)] //[notwin,win]~ ERROR: alignment value: not a power of two
fn f4() {}

#[align(16384)]  //[win]~ ERROR: alignment must not be greater than 8192 bytes for COFF targets
fn f5() {}

fn main() {}
