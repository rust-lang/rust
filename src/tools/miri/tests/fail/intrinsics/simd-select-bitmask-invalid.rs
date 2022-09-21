#![feature(platform_intrinsics, repr_simd)]

extern "platform-intrinsic" {
    fn simd_select_bitmask<M, T>(m: M, yes: T, no: T) -> T;
}

#[repr(simd)]
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
struct i32x2(i32, i32);

fn main() {
    unsafe {
        let x = i32x2(0, 1);
        simd_select_bitmask(0b11111111u8, x, x); //~ERROR: bitmask less than 8 bits long must be filled with 0s for the remaining bits
    }
}
