//! Exact vector sin

use coresimd::simd::*;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.sin.v2f32"]
    fn sin_v2f32(x: f32x2) -> f32x2;
    #[link_name = "llvm.sin.v4f32"]
    fn sin_v4f32(x: f32x4) -> f32x4;
    #[link_name = "llvm.sin.v8f32"]
    fn sin_v8f32(x: f32x8) -> f32x8;
    #[link_name = "llvm.sin.v16f32"]
    fn sin_v16f32(x: f32x16) -> f32x16;
    #[link_name = "llvm.sin.v2f64"]
    fn sin_v2f64(x: f64x2) -> f64x2;
    #[link_name = "llvm.sin.v4f64"]
    fn sin_v4f64(x: f64x4) -> f64x4;
    #[link_name = "llvm.sin.v8f64"]
    fn sin_v8f64(x: f64x8) -> f64x8;
}

pub(crate) trait FloatSin {
    fn sin(self) -> Self;
}

macro_rules! impl_fsin {
    ($id:ident : $fn:ident) => {
        impl FloatSin for $id {
            fn sin(self) -> Self {
                unsafe { $fn(self) }
            }
        }
    };
}

impl_fsin!(f32x2: sin_v2f32);
impl_fsin!(f32x4: sin_v4f32);
impl_fsin!(f32x8: sin_v8f32);
impl_fsin!(f32x16: sin_v16f32);
impl_fsin!(f64x2: sin_v2f64);
impl_fsin!(f64x4: sin_v4f64);
impl_fsin!(f64x8: sin_v8f64);
