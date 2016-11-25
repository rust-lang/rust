pub use self::sse2::*;

mod sse;
mod sse2;

#[repr(simd)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct __m128(f32, f32, f32, f32);

#[repr(simd)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct __m128d(f64, f64);

#[repr(simd)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct __m128i(u64, u64);

#[repr(simd)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct f64x2(f64, f64);

#[repr(simd)]
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
struct u8x16(u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8, u8);
