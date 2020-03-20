#![allow(dead_code)]

#[repr(align(16.0))] //~ ERROR: invalid `repr(align)` attribute: not an unsuffixed integer
                     //~| ERROR: invalid `repr(align)` attribute: not an unsuffixed integer
struct S0(i32);

#[repr(align(15))] //~ ERROR: invalid `repr(align)` attribute: not a power of two
                   //~| ERROR: invalid `repr(align)` attribute: not a power of two
struct S1(i32);

#[repr(align(4294967296))] //~ ERROR: invalid `repr(align)` attribute: larger than 4096
                           //~| ERROR: invalid `repr(align)` attribute: larger than 4096
struct S2(i32);

#[repr(align(4096))] // ok: this is the largest accepted alignment
struct S3(i32);

#[repr(align(16.0))] //~ ERROR: invalid `repr(align)` attribute: not an unsuffixed integer
                     //~| ERROR: invalid `repr(align)` attribute: not an unsuffixed integer
enum E0 { A, B }

#[repr(align(15))] //~ ERROR: invalid `repr(align)` attribute: not a power of two
                   //~| ERROR: invalid `repr(align)` attribute: not a power of two
enum E1 { A, B }

#[repr(align(4294967296))] //~ ERROR: invalid `repr(align)` attribute: larger than 4096
                           //~| ERROR: invalid `repr(align)` attribute: larger than 4096
enum E2 { A, B }

#[repr(align(4096))] // ok: this is the largest accepted alignment
enum E3 { A, B }

fn main() {}
