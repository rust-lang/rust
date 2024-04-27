//@ run-pass
#![feature(repr_simd, intrinsics)]

extern "rust-intrinsic" {
    fn simd_masked_load<M, P, T>(mask: M, pointer: P, values: T) -> T;
    fn simd_masked_store<M, P, T>(mask: M, pointer: P, values: T) -> ();
}

#[derive(Copy, Clone)]
#[repr(simd)]
struct Simd<T, const N: usize>([T; N]);

fn main() {
    unsafe {
        let a = Simd::<u8, 4>([0, 1, 2, 3]);
        let b_src = [4u8, 5, 6, 7];
        let b_default = Simd::<u8, 4>([9; 4]);
        let b: Simd::<u8, 4> = simd_masked_load(
            Simd::<i8, 4>([-1, 0, -1, -1]),
            b_src.as_ptr(),
            b_default
        );

        assert_eq!(&b.0, &[4, 9, 6, 7]);

        let mut output = [u8::MAX; 5];

        simd_masked_store(Simd::<i8, 4>([-1, -1, -1, 0]), output.as_mut_ptr(), a);
        assert_eq!(&output, &[0, 1, 2, u8::MAX, u8::MAX]);
        simd_masked_store(Simd::<i8, 4>([0, -1, -1, 0]), output[1..].as_mut_ptr(), b);
        assert_eq!(&output, &[0, 1, 9, 6, u8::MAX]);
    }
}
