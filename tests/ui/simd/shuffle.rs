// run-pass
// revisions: opt noopt
//[noopt] compile-flags: -Copt-level=0
//[opt] compile-flags: -O
#![feature(repr_simd, platform_intrinsics)]
#![allow(incomplete_features)]
#![feature(adt_const_params)]

extern "platform-intrinsic" {
    fn simd_shuffle<T, I, U>(a: T, b: T, i: I) -> U;
    fn simd_shuffle16<T, U>(x: T, y: T, idx: [u32; 16]) -> U;
}

#[derive(Copy, Clone)]
#[repr(simd)]
struct Simd<T, const N: usize>([T; N]);

pub unsafe fn __shuffle_vector16<const IDX: [u32; 16], T, U>(x: T, y: T) -> U {
    simd_shuffle16(x, y, IDX)
}

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
    // Test that an indirection (via an unnamed constant)
    // through a const generic parameter also works.
    // See https://github.com/rust-lang/rust/issues/113500 for details.
    let a = Simd::<u8, 16>([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let b = Simd::<u8, 16>([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]);
    unsafe {
        __shuffle_vector16::<
            { [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] },
            Simd<u8, 16>,
            Simd<u8, 16>,
        >(a, b);
    }
}
