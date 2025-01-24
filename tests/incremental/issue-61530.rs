#![feature(repr_simd, intrinsics)]

//@ revisions:rpass1 rpass2

#[repr(simd)]
struct I32x2([i32; 2]);

extern "rust-intrinsic" {
    fn simd_shuffle<T, I, U>(x: T, y: T, idx: I) -> U;
}

#[repr(simd)]
struct SimdShuffleIdx<const LEN: usize>([u32; LEN]);

fn main() {
    unsafe {
        const IDX: SimdShuffleIdx<2> = SimdShuffleIdx([0, 0]);
        let _: I32x2 = simd_shuffle(I32x2([1, 2]), I32x2([3, 4]), IDX);
        let _: I32x2 = simd_shuffle(I32x2([1, 2]), I32x2([3, 4]), IDX);
    }
}
