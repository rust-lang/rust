//run-pass
#![feature(repr_simd, platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_shuffle<T, I, U>(a: T, b: T, i: I) -> U;
}

#[derive(Copy, Clone)]
#[repr(simd)]
struct Simd<T, const N: usize>([T; N]);

fn main() {
    const I1: [u32; 4] = [0, 2, 4, 6];
    const I2: [u32; 2] = [1, 5];
    let a = Simd::<u8, 4>([0, 1, 2, 3]);
    let b = Simd::<u8, 4>([4, 5, 6, 7]);
    unsafe {
        let x: Simd<u8, 4> = simd_shuffle(a, b, I1);
        assert_eq!(x.0, [0, 2, 4, 6]);

        let y: Simd<u8, 2> = simd_shuffle(a, b, I2);
        assert_eq!(y.0, [1, 5]);
    }
}
