//! Exact vector cos
#![allow(dead_code)]
use coresimd::simd::*;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.cos.f32"]
    fn cos_f32(x: f32) -> f32;
    #[link_name = "llvm.cos.f64"]
    fn cos_f64(x: f64) -> f64;

    #[link_name = "llvm.cos.v2f32"]
    fn cos_v2f32(x: f32x2) -> f32x2;
    #[link_name = "llvm.cos.v4f32"]
    fn cos_v4f32(x: f32x4) -> f32x4;
    #[link_name = "llvm.cos.v8f32"]
    fn cos_v8f32(x: f32x8) -> f32x8;
    #[link_name = "llvm.cos.v16f32"]
    fn cos_v16f32(x: f32x16) -> f32x16;
    #[link_name = "llvm.cos.v2f64"]
    fn cos_v2f64(x: f64x2) -> f64x2;
    #[link_name = "llvm.cos.v4f64"]
    fn cos_v4f64(x: f64x4) -> f64x4;
    #[link_name = "llvm.cos.v8f64"]
    fn cos_v8f64(x: f64x8) -> f64x8;
}

pub(crate) trait FloatCos {
    fn cos(self) -> Self;
}

trait RawCos {
    fn raw_cos(self) -> Self;
}

impl RawCos for f32 {
    fn raw_cos(self) -> Self {
        unsafe { cos_f32(self) }
    }
}

impl RawCos for f64 {
    fn raw_cos(self) -> Self {
        unsafe { cos_f64(self) }
    }
}


macro_rules! impl_fcos {
    ($id:ident : $fn:ident) => {
        #[cfg(not(target_arch = "s390x"))]
        impl FloatCos for $id {
            fn cos(self) -> Self {
                unsafe { $fn(self) }
            }
        }

        // FIXME: https://github.com/rust-lang-nursery/stdsimd/issues/501
        #[cfg(target_arch = "s390x")]
        impl FloatCos for $id {
            fn cos(self) -> Self {
                let mut v = $id::splat(0.);
                for i in 0..$id::lanes() {
                    v = v.replace(i, self.extract(i).raw_cos())
                }
                v
            }
        }
    };
}

impl_fcos!(f32x2: cos_v2f32);
impl_fcos!(f32x4: cos_v4f32);
impl_fcos!(f32x8: cos_v8f32);
impl_fcos!(f32x16: cos_v16f32);
impl_fcos!(f64x2: cos_v2f64);
impl_fcos!(f64x4: cos_v4f64);
impl_fcos!(f64x8: cos_v8f64);
