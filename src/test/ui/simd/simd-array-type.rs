// run-pass
#![allow(dead_code)]

// pretty-expanded FIXME #23616

#![feature(repr_simd, platform_intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone)]
struct S([i32; 4]);

#[repr(simd)]
#[derive(Copy, Clone)]
struct T<const N: usize>([i32; N]);

extern "platform-intrinsic" {
    fn simd_insert<T, E>(x: T, idx: u32, y: E) -> T;
    fn simd_extract<T, E>(x: T, idx: u32) -> E;
}

pub fn main() {
    let mut s = S([0; 4]);

    unsafe {
        for i in 0_i32..4 {
            s = simd_insert(s, i as u32, i);
        }
        for i in 0_i32..4 {
            assert_eq!(i, simd_extract(s, i as u32));
        }
    }

    let mut t = T::<4>([0; 4]);
    unsafe {
        for i in 0_i32..4 {
            t = simd_insert(t, i as u32, i);
        }
        for i in 0_i32..4 {
            assert_eq!(i, simd_extract(t, i as u32));
        }
    }
}
