#![feature(repr_simd, platform_intrinsics)]

#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct i32x4(i32, i32, i32, i32);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct i32x8(i32, i32, i32, i32,
             i32, i32, i32, i32);

#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct f32x4(f32, f32, f32, f32);
#[repr(simd)]
#[derive(Copy, Clone)]
#[allow(non_camel_case_types)]
struct f32x8(f32, f32, f32, f32,
             f32, f32, f32, f32);


extern "platform-intrinsic" {
    fn simd_cast<T, U>(x: T) -> U;
}

fn main() {
    let x = i32x4(0, 0, 0, 0);

    unsafe {
        simd_cast::<i32, i32>(0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_cast::<i32, i32x4>(0);
        //~^ ERROR expected SIMD input type, found non-SIMD `i32`
        simd_cast::<i32x4, i32>(x);
        //~^ ERROR expected SIMD return type, found non-SIMD `i32`
        simd_cast::<_, i32x8>(x);
//~^ ERROR return type with length 4 (same as input type `i32x4`), found `i32x8` with length 8
    }
}
