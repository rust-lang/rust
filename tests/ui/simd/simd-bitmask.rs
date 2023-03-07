//run-pass
//ignore-endian-big behavior of simd_select_bitmask is endian-specific
#![feature(repr_simd, platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_bitmask<T, U>(v: T) -> U;
    fn simd_select_bitmask<T, U>(m: T, a: U, b: U) -> U;
}

#[derive(Copy, Clone)]
#[repr(simd)]
struct Simd<T, const N: usize>([T; N]);

fn main() {
    unsafe {
        let v = Simd::<i8, 4>([-1, 0, -1, 0]);
        let i: u8 = simd_bitmask(v);
        let a: [u8; 1] = simd_bitmask(v);

        assert_eq!(i, 0b0101);
        assert_eq!(a, [0b0101]);

        let v = Simd::<i8, 16>([0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0]);
        let i: u16 = simd_bitmask(v);
        let a: [u8; 2] = simd_bitmask(v);

        assert_eq!(i, 0b0101000000001100);
        assert_eq!(a, [0b1100, 0b01010000]);
    }

    unsafe {
        let a = Simd::<i32, 8>([0, 1, 2, 3, 4, 5, 6, 7]);
        let b = Simd::<i32, 8>([8, 9, 10, 11, 12, 13, 14, 15]);
        let e = [0, 9, 2, 11, 12, 13, 14, 15];

        let r = simd_select_bitmask(0b0101u8, a, b);
        assert_eq!(r.0, e);

        let r = simd_select_bitmask([0b0101u8], a, b);
        assert_eq!(r.0, e);

        let a = Simd::<i32, 16>([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let b = Simd::<i32, 16>([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]);
        let e = [16, 17, 2, 3, 20, 21, 22, 23, 24, 25, 26, 27, 12, 29, 14, 31];

        let r = simd_select_bitmask(0b0101000000001100u16, a, b);
        assert_eq!(r.0, e);

        let r = simd_select_bitmask([0b1100u8, 0b01010000u8], a, b);
        assert_eq!(r.0, e);
    }
}
