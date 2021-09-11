//run-pass
#![feature(repr_simd, platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_bitmask<T, U>(v: T) -> U;
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
}
