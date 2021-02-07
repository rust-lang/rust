// run-pass
#![allow(deprecated)]


#![feature(repr_simd)]
#![allow(non_camel_case_types)]

use std::mem;

/// `T` should satisfy `size_of T (mod min_align_of T) === 0` to be stored at `Vec<T>` properly
/// Please consult the issue #20460
fn check<T>() {
    assert_eq!(mem::size_of::<T>() % mem::min_align_of::<T>(), 0);
    assert_eq!(mem::size_of::<T>() % mem::min_align_of::<T>(), 0);
    assert_eq!(mem::size_of::<T>() % mem::min_align_of::<T>(), 0);
}

#[repr(simd)]
struct U8<const N: usize>([u8; N]);

#[repr(simd)]
struct I16<const N: usize>([i16; N]);

#[repr(simd)]
struct F32<const N: usize>([f32; N]);

#[repr(simd)]
struct Usize<const N: usize>([usize; N]);

#[repr(simd)]
struct Isize<const N: usize>([isize; N]);

fn main() {
    check::<U8<2>>();
    check::<U8<4>>();
    check::<U8<8>>();

    check::<I16<2>>();
    check::<I16<4>>();
    check::<I16<8>>();

    check::<F32<2>>();
    check::<F32<4>>();
    check::<F32<8>>();

    check::<Usize<2>>();
    check::<Usize<4>>();
    check::<Usize<8>>();

    check::<Isize<2>>();
    check::<Isize<4>>();
    check::<Isize<8>>();
}
