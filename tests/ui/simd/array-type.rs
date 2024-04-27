//@ run-pass
#![allow(dead_code)]

//@ pretty-expanded FIXME #23616

#![feature(repr_simd, intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone)]
struct S([i32; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct T<const N: usize>([i32; N]);

extern "rust-intrinsic" {
    fn simd_insert<T, E>(x: T, idx: u32, y: E) -> T;
    fn simd_extract<T, E>(x: T, idx: u32) -> E;
}

pub fn main() {
    let mut s = S([0; 4]);

    unsafe {
        s = simd_insert(s, 3, 3);
        assert_eq!(3, simd_extract(s, 3));
    }

    let mut t = T::<4>([0; 4]);
    unsafe {
        t = simd_insert(t, 3, 3);
        assert_eq!(3, simd_extract(t, 3));
    }
}
