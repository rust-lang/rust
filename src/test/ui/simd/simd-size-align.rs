// run-pass
#![allow(deprecated)]


#![feature(repr_simd)]
#![allow(non_camel_case_types)]

use std::mem;

/// `T` should satisfy `size_of T (mod min_align_of T) === 0` to be stored at `Vec<T>` properly
/// Please consult the issue #20460
fn check<T>() {
    assert_eq!(mem::size_of::<T>() % mem::min_align_of::<T>(), 0)
}

fn main() {
    check::<u8x2>();
    check::<u8x3>();
    check::<u8x4>();
    check::<u8x5>();
    check::<u8x6>();
    check::<u8x7>();
    check::<u8x8>();

    check::<i16x2>();
    check::<i16x3>();
    check::<i16x4>();
    check::<i16x5>();
    check::<i16x6>();
    check::<i16x7>();
    check::<i16x8>();

    check::<f32x2>();
    check::<f32x3>();
    check::<f32x4>();
    check::<f32x5>();
    check::<f32x6>();
    check::<f32x7>();
    check::<f32x8>();

    check::<usizex2>();
    check::<usizex3>();
    check::<usizex4>();
    check::<usizex5>();
    check::<usizex6>();
    check::<usizex7>();
    check::<usizex8>();

    check::<isizex2>();
    check::<isizex3>();
    check::<isizex4>();
    check::<isizex5>();
    check::<isizex6>();
    check::<isizex7>();
    check::<isizex8>();
}

#[repr(simd)] struct u8x2(u8, u8);
#[repr(simd)] struct u8x3(u8, u8, u8);
#[repr(simd)] struct u8x4(u8, u8, u8, u8);
#[repr(simd)] struct u8x5(u8, u8, u8, u8, u8);
#[repr(simd)] struct u8x6(u8, u8, u8, u8, u8, u8);
#[repr(simd)] struct u8x7(u8, u8, u8, u8, u8, u8, u8);
#[repr(simd)] struct u8x8(u8, u8, u8, u8, u8, u8, u8, u8);

#[repr(simd)] struct i16x2(i16, i16);
#[repr(simd)] struct i16x3(i16, i16, i16);
#[repr(simd)] struct i16x4(i16, i16, i16, i16);
#[repr(simd)] struct i16x5(i16, i16, i16, i16, i16);
#[repr(simd)] struct i16x6(i16, i16, i16, i16, i16, i16);
#[repr(simd)] struct i16x7(i16, i16, i16, i16, i16, i16, i16);
#[repr(simd)] struct i16x8(i16, i16, i16, i16, i16, i16, i16, i16);

#[repr(simd)] struct f32x2(f32, f32);
#[repr(simd)] struct f32x3(f32, f32, f32);
#[repr(simd)] struct f32x4(f32, f32, f32, f32);
#[repr(simd)] struct f32x5(f32, f32, f32, f32, f32);
#[repr(simd)] struct f32x6(f32, f32, f32, f32, f32, f32);
#[repr(simd)] struct f32x7(f32, f32, f32, f32, f32, f32, f32);
#[repr(simd)] struct f32x8(f32, f32, f32, f32, f32, f32, f32, f32);

#[repr(simd)] struct usizex2(usize, usize);
#[repr(simd)] struct usizex3(usize, usize, usize);
#[repr(simd)] struct usizex4(usize, usize, usize, usize);
#[repr(simd)] struct usizex5(usize, usize, usize, usize, usize);
#[repr(simd)] struct usizex6(usize, usize, usize, usize, usize, usize);
#[repr(simd)] struct usizex7(usize, usize, usize, usize, usize, usize, usize);
#[repr(simd)] struct usizex8(usize, usize, usize, usize, usize, usize, usize, usize);

#[repr(simd)] struct isizex2(isize, isize);
#[repr(simd)] struct isizex3(isize, isize, isize);
#[repr(simd)] struct isizex4(isize, isize, isize, isize);
#[repr(simd)] struct isizex5(isize, isize, isize, isize, isize);
#[repr(simd)] struct isizex6(isize, isize, isize, isize, isize, isize);
#[repr(simd)] struct isizex7(isize, isize, isize, isize, isize, isize, isize);
#[repr(simd)] struct isizex8(isize, isize, isize, isize, isize, isize, isize, isize);
