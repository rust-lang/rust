//@ run-pass

#![feature(repr_simd, core_intrinsics)]

use std::intrinsics::simd::simd_as;

#[derive(Copy, Clone)]
#[repr(simd)]
struct V<T>([T; 2]);
impl<T> V<T> {
    fn to_array(self) -> [T; 2] { unsafe { std::intrinsics::transmute_unchecked(self) } }
}

fn main() {
    unsafe {
        let u = V::<u32>([u32::MIN, u32::MAX]);
        let i: V<i16> = simd_as(u);
        assert_eq!(i.to_array()[0], u.to_array()[0] as i16);
        assert_eq!(i.to_array()[1], u.to_array()[1] as i16);
    }

    unsafe {
        let f = V::<f32>([f32::MIN, f32::MAX]);
        let i: V<i16> = simd_as(f);
        assert_eq!(i.to_array()[0], f.to_array()[0] as i16);
        assert_eq!(i.to_array()[1], f.to_array()[1] as i16);
    }

    unsafe {
        let f = V::<f32>([f32::MIN, f32::MAX]);
        let u: V<u8> = simd_as(f);
        assert_eq!(u.to_array()[0], f.to_array()[0] as u8);
        assert_eq!(u.to_array()[1], f.to_array()[1] as u8);
    }

    unsafe {
        let f = V::<f64>([f64::MIN, f64::MAX]);
        let i: V<isize> = simd_as(f);
        assert_eq!(i.to_array()[0], f.to_array()[0] as isize);
        assert_eq!(i.to_array()[1], f.to_array()[1] as isize);
    }

    unsafe {
        let f = V::<f64>([f64::MIN, f64::MAX]);
        let u: V<usize> = simd_as(f);
        assert_eq!(u.to_array()[0], f.to_array()[0] as usize);
        assert_eq!(u.to_array()[1], f.to_array()[1] as usize);
    }
}
