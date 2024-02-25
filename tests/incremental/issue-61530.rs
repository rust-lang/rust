#![feature(repr_simd, platform_intrinsics)]

//@ revisions:rpass1 rpass2

#[repr(simd)]
struct I32x2(i32, i32);

extern "platform-intrinsic" {
    fn simd_shuffle<T, I, U>(x: T, y: T, idx: I) -> U;
}

fn main() {
    unsafe {
        const IDX: [u32; 2] = [0, 0];
        let _: I32x2 = simd_shuffle(I32x2(1, 2), I32x2(3, 4), IDX);
        let _: I32x2 = simd_shuffle(I32x2(1, 2), I32x2(3, 4), IDX);
    }
}
