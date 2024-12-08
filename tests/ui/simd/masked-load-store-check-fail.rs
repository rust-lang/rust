//@ check-fail
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
        let mut arr = [4u8, 5, 6, 7];
        let default = Simd::<u8, 4>([9; 4]);

        let _x: Simd<u8, 2> = simd_masked_load(
            Simd::<i8, 4>([-1, 0, -1, -1]),
            arr.as_ptr(),
            Simd::<u8, 4>([9; 4])
        );
        //~^^ ERROR mismatched types

        let _x: Simd<u32, 4> = simd_masked_load(
            Simd::<u8, 4>([1, 0, 1, 1]),
            arr.as_ptr(),
            default
        );
        //~^^ ERROR mismatched types
    }
}
