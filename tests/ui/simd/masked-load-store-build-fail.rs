//@ build-fail
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

        simd_masked_load(
            Simd::<i8, 8>([-1, 0, -1, -1, 0, 0, 0, 0]),
            arr.as_ptr(),
            default
        );
        //~^^^^^ ERROR expected third argument with length 8 (same as input type `Simd<i8, 8>`), found `Simd<u8, 4>` with length 4

        simd_masked_load(
            Simd::<i8, 4>([-1, 0, -1, -1]),
            arr.as_ptr() as *const i8,
            default
        );
        //~^^^^^ ERROR expected element type `u8` of second argument `*const i8` to be a pointer to the element type `u8` of the first argument `Simd<u8, 4>`, found `u8` != `*_ u8`

        simd_masked_load(
            Simd::<i8, 4>([-1, 0, -1, -1]),
            arr.as_ptr(),
            Simd::<u32, 4>([9; 4])
        );
        //~^^^^^ ERROR expected element type `u32` of second argument `*const u8` to be a pointer to the element type `u32` of the first argument `Simd<u32, 4>`, found `u32` != `*_ u32`

        simd_masked_load(
            Simd::<u8, 4>([1, 0, 1, 1]),
            arr.as_ptr(),
            default
        );
        //~^^^^^ ERROR expected element type `u8` of third argument `Simd<u8, 4>` to be a signed integer type

        simd_masked_store(
            Simd([-1i8; 4]),
            arr.as_ptr(),
            Simd([5u32; 4])
        );
        //~^^^^^ ERROR expected element type `u32` of second argument `*const u8` to be a pointer to the element type `u32` of the first argument `Simd<u32, 4>`, found `u32` != `*mut u32`

        simd_masked_store(
            Simd([-1i8; 4]),
            arr.as_ptr(),
            Simd([5u8; 4])
        );
        //~^^^^^ ERROR expected element type `u8` of second argument `*const u8` to be a pointer to the element type `u8` of the first argument `Simd<u8, 4>`, found `u8` != `*mut u8`

        simd_masked_store(
            Simd([-1i8; 4]),
            arr.as_mut_ptr(),
            Simd([5u8; 2])
        );
        //~^^^^^ ERROR expected third argument with length 4 (same as input type `Simd<i8, 4>`), found `Simd<u8, 2>` with length 2

        simd_masked_store(
            Simd([1u32; 4]),
            arr.as_mut_ptr(),
            Simd([5u8; 4])
        );
        //~^^^^^ ERROR expected element type `u8` of third argument `Simd<u32, 4>` to be a signed integer type
    }
}
