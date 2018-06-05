//! Exact vector square-root

use coresimd::simd::*;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.sqrt.v2f32"]
    fn sqrt_v2f32(x: f32x2) -> f32x2;
    #[link_name = "llvm.sqrt.v4f32"]
    fn sqrt_v4f32(x: f32x4) -> f32x4;
    #[link_name = "llvm.sqrt.v8f32"]
    fn sqrt_v8f32(x: f32x8) -> f32x8;
    #[link_name = "llvm.sqrt.v16f32"]
    fn sqrt_v16f32(x: f32x16) -> f32x16;
    #[link_name = "llvm.sqrt.v2f64"]
    fn sqrt_v2f64(x: f64x2) -> f64x2;
    #[link_name = "llvm.sqrt.v4f64"]
    fn sqrt_v4f64(x: f64x4) -> f64x4;
    #[link_name = "llvm.sqrt.v8f64"]
    fn sqrt_v8f64(x: f64x8) -> f64x8;
}

pub(crate) trait FloatSqrt {
    fn sqrt(self) -> Self;
}

macro_rules! impl_fsqrt {
    ($id:ident : $fn:ident) => {
        impl FloatSqrt for $id {
            fn sqrt(self) -> Self {
                unsafe { $fn(self) }
            }
        }
    };
}

impl_fsqrt!(f32x2: sqrt_v2f32);
impl_fsqrt!(f32x4: sqrt_v4f32);
impl_fsqrt!(f32x8: sqrt_v8f32);
impl_fsqrt!(f32x16: sqrt_v16f32);
impl_fsqrt!(f64x2: sqrt_v2f64);
impl_fsqrt!(f64x4: sqrt_v4f64);
impl_fsqrt!(f64x8: sqrt_v8f64);
