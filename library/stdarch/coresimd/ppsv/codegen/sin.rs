//! Exact vector sin
#![allow(dead_code)]
use coresimd::simd::*;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.sin.f32"]
    fn sin_f32(x: f32) -> f32;
    #[link_name = "llvm.sin.f64"]
    fn sin_f64(x: f64) -> f64;

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

trait RawSin {
    fn raw_sin(self) -> Self;
}

impl RawSin for f32 {
    fn raw_sin(self) -> Self {
        unsafe { sin_f32(self) }
    }
}

impl RawSin for f64 {
    fn raw_sin(self) -> Self {
        unsafe { sin_f64(self) }
    }
}

macro_rules! impl_fsin {
    ($id:ident : $fn:ident) => {
        #[cfg(not(target_arch = "s390x"))]
        impl FloatSin for $id {
            fn sin(self) -> Self {
                unsafe { $fn(self) }
            }
        }

        // FIXME: https://github.com/rust-lang-nursery/stdsimd/issues/501
        #[cfg(target_arch = "s390x")]
        impl FloatSin for $id {
            fn sin(self) -> Self {
                let mut v = $id::splat(0.);
                for i in 0..$id::lanes() {
                    v = v.replace(i, self.extract(i).raw_sin())
                }
                v
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
