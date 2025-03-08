//@ run-pass
#![feature(repr_simd, core_intrinsics)]

use std::intrinsics::simd::simd_cast;

#[derive(Copy, Clone)]
#[repr(simd)]
struct V<T>([T; 4]);
impl<T> V<T> {
    fn to_array(self) -> [T; 4] { unsafe { std::intrinsics::transmute_unchecked(self) } }
}

fn main() {
    let u = V::<usize>([0, 1, 2, 3]);
    let uu32: V<u32> = unsafe { simd_cast(u) };
    let ui64: V<i64> = unsafe { simd_cast(u) };

    for (u, (uu32, ui64)) in u
        .to_array()
        .iter()
        .zip(uu32.to_array().iter().zip(ui64.to_array().iter()))
    {
        assert_eq!(*u as u32, *uu32);
        assert_eq!(*u as i64, *ui64);
    }
}
