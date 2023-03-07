// run-pass
// ignore-emscripten FIXME(#45351) hits an LLVM assert

#![feature(repr_simd, platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_cast<T, U>(x: T) -> U;
}

use std::cmp::{max, min};

#[derive(Copy, Clone)]
#[repr(simd)]
struct V<T>([T; 2]);

fn main() {
    unsafe {
        let u = V::<u32>([i16::MIN as u32, i16::MAX as u32]);
        let i: V<i16> = simd_cast(u);
        assert_eq!(i.0[0], u.0[0] as i16);
        assert_eq!(i.0[1], u.0[1] as i16);
    }

    unsafe {
        let f = V::<f32>([i16::MIN as f32, i16::MAX as f32]);
        let i: V<i16> = simd_cast(f);
        assert_eq!(i.0[0], f.0[0] as i16);
        assert_eq!(i.0[1], f.0[1] as i16);
    }

    unsafe {
        let f = V::<f32>([u8::MIN as f32, u8::MAX as f32]);
        let u: V<u8> = simd_cast(f);
        assert_eq!(u.0[0], f.0[0] as u8);
        assert_eq!(u.0[1], f.0[1] as u8);
    }

    unsafe {
        // We would like to do isize::MIN..=isize::MAX, but those values are not representable in
        // an f64, so we clamp to the range of an i32 to prevent running into UB.
        let f = V::<f64>([
            max(isize::MIN, i32::MIN as isize) as f64,
            min(isize::MAX, i32::MAX as isize) as f64,
        ]);
        let i: V<isize> = simd_cast(f);
        assert_eq!(i.0[0], f.0[0] as isize);
        assert_eq!(i.0[1], f.0[1] as isize);
    }

    unsafe {
        let f = V::<f64>([
            max(usize::MIN, u32::MIN as usize) as f64,
            min(usize::MAX, u32::MAX as usize) as f64,
        ]);
        let u: V<usize> = simd_cast(f);
        assert_eq!(u.0[0], f.0[0] as usize);
        assert_eq!(u.0[1], f.0[1] as usize);
    }
}
