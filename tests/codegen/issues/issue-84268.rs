// compile-flags: -O --crate-type=rlib
#![feature(platform_intrinsics, repr_simd)]

extern "platform-intrinsic" {
    fn simd_fabs<T>(x: T) -> T;
    fn simd_eq<T, U>(x: T, y: T) -> U;
}

#[repr(simd)]
pub struct V([f32; 4]);

#[repr(simd)]
pub struct M([i32; 4]);

#[no_mangle]
// CHECK-LABEL: @is_infinite
pub fn is_infinite(v: V) -> M {
    // CHECK: fabs
    // CHECK: cmp oeq
    unsafe {
        simd_eq(simd_fabs(v), V([f32::INFINITY; 4]))
    }
}
