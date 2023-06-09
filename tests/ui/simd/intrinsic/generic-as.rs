// run-pass

#![feature(repr_simd, platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_as<T, U>(x: T) -> U;
}

#[derive(Copy, Clone)]
#[repr(simd)]
struct V<T>([T; 2]);

fn main() {
    unsafe {
        let u = V::<u32>([u32::MIN, u32::MAX]);
        let i: V<i16> = simd_as(u);
        assert_eq!(i.0[0], u.0[0] as i16);
        assert_eq!(i.0[1], u.0[1] as i16);
    }

    unsafe {
        let f = V::<f32>([f32::MIN, f32::MAX]);
        let i: V<i16> = simd_as(f);
        assert_eq!(i.0[0], f.0[0] as i16);
        assert_eq!(i.0[1], f.0[1] as i16);
    }

    unsafe {
        let f = V::<f32>([f32::MIN, f32::MAX]);
        let u: V<u8> = simd_as(f);
        assert_eq!(u.0[0], f.0[0] as u8);
        assert_eq!(u.0[1], f.0[1] as u8);
    }

    unsafe {
        let f = V::<f64>([f64::MIN, f64::MAX]);
        let i: V<isize> = simd_as(f);
        assert_eq!(i.0[0], f.0[0] as isize);
        assert_eq!(i.0[1], f.0[1] as isize);
    }

    unsafe {
        let f = V::<f64>([f64::MIN, f64::MAX]);
        let u: V<usize> = simd_as(f);
        assert_eq!(u.0[0], f.0[0] as usize);
        assert_eq!(u.0[1], f.0[1] as usize);
    }
}
