//@ run-pass

#![feature(repr_simd, core_intrinsics)]

use std::intrinsics::simd::simd_cast;

use std::cmp::{max, min};

#[derive(Copy, Clone)]
#[repr(simd)]
struct V<T>([T; 2]);
impl<T> V<T> {
    fn to_array(self) -> [T; 2] { unsafe { std::intrinsics::transmute_unchecked(self) } }
}

fn main() {
    unsafe {
        let u = V::<u32>([i16::MIN as u32, i16::MAX as u32]);
        let i: V<i16> = simd_cast(u);
        assert_eq!(i.to_array()[0], u.to_array()[0] as i16);
        assert_eq!(i.to_array()[1], u.to_array()[1] as i16);
    }

    unsafe {
        let f = V::<f32>([i16::MIN as f32, i16::MAX as f32]);
        let i: V<i16> = simd_cast(f);
        assert_eq!(i.to_array()[0], f.to_array()[0] as i16);
        assert_eq!(i.to_array()[1], f.to_array()[1] as i16);
    }

    unsafe {
        let f = V::<f32>([u8::MIN as f32, u8::MAX as f32]);
        let u: V<u8> = simd_cast(f);
        assert_eq!(u.to_array()[0], f.to_array()[0] as u8);
        assert_eq!(u.to_array()[1], f.to_array()[1] as u8);
    }

    unsafe {
        // We would like to do isize::MIN..=isize::MAX, but those values are not representable in
        // an f64, so we clamp to the range of an i32 to prevent running into UB.
        let f = V::<f64>([
            max(isize::MIN, i32::MIN as isize) as f64,
            min(isize::MAX, i32::MAX as isize) as f64,
        ]);
        let i: V<isize> = simd_cast(f);
        assert_eq!(i.to_array()[0], f.to_array()[0] as isize);
        assert_eq!(i.to_array()[1], f.to_array()[1] as isize);
    }

    unsafe {
        let f = V::<f64>([
            max(usize::MIN, u32::MIN as usize) as f64,
            min(usize::MAX, u32::MAX as usize) as f64,
        ]);
        let u: V<usize> = simd_cast(f);
        assert_eq!(u.to_array()[0], f.to_array()[0] as usize);
        assert_eq!(u.to_array()[1], f.to_array()[1] as usize);
    }
}
