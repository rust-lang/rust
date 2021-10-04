// build-fail

// Test that the simd_shuffle intrinsic produces ok-ish error
// messages when misused.

#![feature(repr_simd, platform_intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct Simd<T, const N: usize>([T; N]);

extern "platform-intrinsic" {
    fn simd_shuffle<T, I, U>(a: T, b: T, i: I) -> U;
}

fn main() {
    const I: [u32; 2] = [0; 2];
    const I2: [f32; 2] = [0.; 2];
    let v = Simd::<u32, 4>([0; 4]);

    unsafe {
        let _: Simd<u32, 2> = simd_shuffle(v, v, I);

        let _: Simd<u32, 4> = simd_shuffle(v, v, I);
        //~^ ERROR invalid monomorphization of `simd_shuffle` intrinsic

        let _: Simd<f32, 2> = simd_shuffle(v, v, I);
        //~^ ERROR invalid monomorphization of `simd_shuffle` intrinsic

        let _: Simd<u32, 2> = simd_shuffle(v, v, I2);
        //~^ ERROR invalid monomorphization of `simd_shuffle` intrinsic
    }
}
