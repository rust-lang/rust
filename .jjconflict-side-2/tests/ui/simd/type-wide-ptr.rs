//@ build-fail

#![feature(repr_simd)]

#[repr(simd)]
struct S([*mut [u8]; 4]);

fn main() {
    let _v: Option<S> = None;
}

//~? ERROR monomorphising SIMD type `S` with a non-primitive-scalar (integer/float/pointer) element type `*mut [u8]`
