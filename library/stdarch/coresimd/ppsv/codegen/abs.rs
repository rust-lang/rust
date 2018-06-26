//! Vector absolute value
#![allow(dead_code)]
use coresimd::simd::*;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.fabs.f32"]
    fn abs_f32(x: f32) -> f32;
    #[link_name = "llvm.fabs.f64"]
    fn abs_f64(x: f64) -> f64;

    #[link_name = "llvm.fabs.v2f32"]
    fn abs_v2f32(x: f32x2) -> f32x2;
    #[link_name = "llvm.fabs.v4f32"]
    fn abs_v4f32(x: f32x4) -> f32x4;
    #[link_name = "llvm.fabs.v8f32"]
    fn abs_v8f32(x: f32x8) -> f32x8;
    #[link_name = "llvm.fabs.v16f32"]
    fn abs_v16f32(x: f32x16) -> f32x16;
    #[link_name = "llvm.fabs.v2f64"]
    fn abs_v2f64(x: f64x2) -> f64x2;
    #[link_name = "llvm.fabs.v4f64"]
    fn abs_v4f64(x: f64x4) -> f64x4;
    #[link_name = "llvm.fabs.v8f64"]
    fn abs_v8f64(x: f64x8) -> f64x8;
}

pub(crate) trait FloatAbs {
    fn abs(self) -> Self;
}

trait RawAbs {
    fn raw_abs(self) -> Self;
}

impl RawAbs for f32 {
    fn raw_abs(self) -> Self {
        unsafe { abs_f32(self) }
    }
}

impl RawAbs for f64 {
    fn raw_abs(self) -> Self {
        unsafe { abs_f64(self) }
    }
}


macro_rules! impl_fabs {
    ($id:ident : $fn:ident) => {
        #[cfg(not(target_arch = "s390x"))]
        impl FloatAbs for $id {
            fn abs(self) -> Self {
                unsafe { $fn(self) }
            }
        }
        // FIXME: https://github.com/rust-lang-nursery/stdsimd/issues/501
        #[cfg(target_arch = "s390x")]
        impl FloatAbs for $id {
            fn abs(self) -> Self {
                let mut v = $id::splat(0.);
                for i in 0..$id::lanes() {
                    v = v.replace(i, self.extract(i).raw_abs())
                }
                v
            }
        }
    };
}

impl_fabs!(f32x2: abs_v2f32);
impl_fabs!(f32x4: abs_v4f32);
impl_fabs!(f32x8: abs_v8f32);
impl_fabs!(f32x16: abs_v16f32);
impl_fabs!(f64x2: abs_v2f64);
impl_fabs!(f64x4: abs_v4f64);
impl_fabs!(f64x8: abs_v8f64);
