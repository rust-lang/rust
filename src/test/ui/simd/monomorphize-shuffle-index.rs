//run-pass
#![feature(repr_simd, platform_intrinsics)]

extern "platform-intrinsic" {
    fn simd_shuffle<T, I, U>(a: T, b: T, i: I) -> U;
}

#[derive(Copy, Clone)]
#[repr(simd)]
struct Simd<T, const N: usize>([T; N]);

trait Shuffle<const N: usize> {
    const I: [u32; N];

    unsafe fn shuffle<T, const M: usize>(&self, a: Simd<T, M>, b: Simd<T, M>) -> Simd<T, N> {
        simd_shuffle(a, b, Self::I)
    }
}

fn main() {
    struct I1;
    impl Shuffle<4> for I1 {
        const I: [u32; 4] = [0, 2, 4, 6];
    }

    struct I2;
    impl Shuffle<2> for I2 {
        const I: [u32; 2] = [1, 5];
    }

    let a = Simd::<u8, 4>([0, 1, 2, 3]);
    let b = Simd::<u8, 4>([4, 5, 6, 7]);
    unsafe {
        let x: Simd<u8, 4> = I1.shuffle(a, b);
        assert_eq!(x.0, [0, 2, 4, 6]);

        let y: Simd<u8, 2> = I2.shuffle(a, b);
        assert_eq!(y.0, [1, 5]);
    }
}
