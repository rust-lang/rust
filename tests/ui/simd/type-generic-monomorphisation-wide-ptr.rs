//@ build-fail

#![feature(repr_simd)]

#[repr(simd)]
struct S<T>([T; 4]);

fn main() {
    let _v: Option<S<*mut [u8]>> = None;
}

//~? ERROR monomorphising SIMD type `S<*mut [u8]>` with a non-primitive-scalar (integer/float/pointer) element type `*mut [u8]`
