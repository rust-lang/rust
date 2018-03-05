//! Code generation for the max reduction.
use ::coresimd::simd::*;

/// LLVM intrinsics used in the max reduction
#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.experimental.vector.reduce.smax.i8.v2i8"]
    fn reduce_max_i8x2(x: i8x2) -> i8;
    #[link_name = "llvm.experimental.vector.reduce.umax.u8.v2u8"]
    fn reduce_max_u8x2(x: u8x2) -> u8;
    #[link_name = "llvm.experimental.vector.reduce.smax.i16.v2i16"]
    fn reduce_max_i16x2(x: i16x2) -> i16;
    #[link_name = "llvm.experimental.vector.reduce.umax.u16.v2u16"]
    fn reduce_max_u16x2(x: u16x2) -> u16;
    #[link_name = "llvm.experimental.vector.reduce.smax.i32.v2i32"]
    fn reduce_max_i32x2(x: i32x2) -> i32;
    #[link_name = "llvm.experimental.vector.reduce.umax.u32.v2u32"]
    fn reduce_max_u32x2(x: u32x2) -> u32;
    #[link_name = "llvm.experimental.vector.reduce.smax.i64.v2i64"]
    fn reduce_max_i64x2(x: i64x2) -> i64;
    #[link_name = "llvm.experimental.vector.reduce.umax.u64.v2u64"]
    fn reduce_max_u64x2(x: u64x2) -> u64;
    #[link_name = "llvm.experimental.vector.reduce.smax.i8.v4i8"]
    fn reduce_max_i8x4(x: i8x4) -> i8;
    #[link_name = "llvm.experimental.vector.reduce.umax.u8.v4u8"]
    fn reduce_max_u8x4(x: u8x4) -> u8;
    #[link_name = "llvm.experimental.vector.reduce.smax.i16.v4i16"]
    fn reduce_max_i16x4(x: i16x4) -> i16;
    #[link_name = "llvm.experimental.vector.reduce.umax.u16.v4u16"]
    fn reduce_max_u16x4(x: u16x4) -> u16;
    #[link_name = "llvm.experimental.vector.reduce.smax.i32.v4i32"]
    fn reduce_max_i32x4(x: i32x4) -> i32;
    #[link_name = "llvm.experimental.vector.reduce.umax.u32.v4u32"]
    fn reduce_max_u32x4(x: u32x4) -> u32;
    #[link_name = "llvm.experimental.vector.reduce.smax.i64.v4i64"]
    fn reduce_max_i64x4(x: i64x4) -> i64;
    #[link_name = "llvm.experimental.vector.reduce.umax.u64.v4u64"]
    fn reduce_max_u64x4(x: u64x4) -> u64;
    #[link_name = "llvm.experimental.vector.reduce.smax.i8.v8i8"]
    fn reduce_max_i8x8(x: i8x8) -> i8;
    #[link_name = "llvm.experimental.vector.reduce.umax.u8.v8u8"]
    fn reduce_max_u8x8(x: u8x8) -> u8;
    #[link_name = "llvm.experimental.vector.reduce.smax.i16.v8i16"]
    fn reduce_max_i16x8(x: i16x8) -> i16;
    #[link_name = "llvm.experimental.vector.reduce.umax.u16.v8u16"]
    fn reduce_max_u16x8(x: u16x8) -> u16;
    #[link_name = "llvm.experimental.vector.reduce.smax.i32.v8i32"]
    fn reduce_max_i32x8(x: i32x8) -> i32;
    #[link_name = "llvm.experimental.vector.reduce.umax.u32.v8u32"]
    fn reduce_max_u32x8(x: u32x8) -> u32;
    #[link_name = "llvm.experimental.vector.reduce.smax.i64.v8i64"]
    fn reduce_max_i64x8(x: i64x8) -> i64;
    #[link_name = "llvm.experimental.vector.reduce.umax.u64.v8u64"]
    fn reduce_max_u64x8(x: u64x8) -> u64;
    #[link_name = "llvm.experimental.vector.reduce.smax.i8.v16i8"]
    fn reduce_max_i8x16(x: i8x16) -> i8;
    #[link_name = "llvm.experimental.vector.reduce.umax.u8.v16u8"]
    fn reduce_max_u8x16(x: u8x16) -> u8;
    #[link_name = "llvm.experimental.vector.reduce.smax.i16.v16i16"]
    fn reduce_max_i16x16(x: i16x16) -> i16;
    #[link_name = "llvm.experimental.vector.reduce.umax.u16.v16u16"]
    fn reduce_max_u16x16(x: u16x16) -> u16;
    #[link_name = "llvm.experimental.vector.reduce.smax.i32.v16i32"]
    fn reduce_max_i32x16(x: i32x16) -> i32;
    #[link_name = "llvm.experimental.vector.reduce.umax.u32.v16u32"]
    fn reduce_max_u32x16(x: u32x16) -> u32;
    #[link_name = "llvm.experimental.vector.reduce.smax.i8.v32i8"]
    fn reduce_max_i8x32(x: i8x32) -> i8;
    #[link_name = "llvm.experimental.vector.reduce.umax.u8.v32u8"]
    fn reduce_max_u8x32(x: u8x32) -> u8;
    #[link_name = "llvm.experimental.vector.reduce.smax.i16.v32i16"]
    fn reduce_max_i16x32(x: i16x32) -> i16;
    #[link_name = "llvm.experimental.vector.reduce.umax.u16.v32u16"]
    fn reduce_max_u16x32(x: u16x32) -> u16;
    #[link_name = "llvm.experimental.vector.reduce.smax.i8.v64i8"]
    fn reduce_max_i8x64(x: i8x64) -> i8;
    #[link_name = "llvm.experimental.vector.reduce.umax.u8.v64u8"]
    fn reduce_max_u8x64(x: u8x64) -> u8;
    #[link_name = "llvm.experimental.vector.reduce.fmax.f32.v2f32"]
    fn reduce_fmax_f32x2(x: f32x2) -> f32;
    #[link_name = "llvm.experimental.vector.reduce.fmax.f64.v2f64"]
    fn reduce_fmax_f64x2(x: f64x2) -> f64;
    #[link_name = "llvm.experimental.vector.reduce.fmax.f32.v4f32"]
    fn reduce_fmax_f32x4(x: f32x4) -> f32;
    #[link_name = "llvm.experimental.vector.reduce.fmax.f64.v4f64"]
    fn reduce_fmax_f64x4(x: f64x4) -> f64;
    #[link_name = "llvm.experimental.vector.reduce.fmax.f32.v8f32"]
    fn reduce_fmax_f32x8(x: f32x8) -> f32;
    #[link_name = "llvm.experimental.vector.reduce.fmax.f64.v8f64"]
    fn reduce_fmax_f64x8(x: f64x8) -> f64;
    #[link_name = "llvm.experimental.vector.reduce.fmax.f32.v16f32"]
    fn reduce_fmax_f32x16(x: f32x16) -> f32;
}

/// Reduction: horizontal max of the vector elements.
#[cfg_attr(feature = "cargo-clippy", allow(stutter))]
pub trait ReduceMax {
    /// Result type of the reduction.
    type Acc;
    /// Computes the horizontal max of the vector elements.
    fn reduce_max(self) -> Self::Acc;
}

macro_rules! red_max {
    ($id:ident, $elem_ty:ident, $llvm_intr:ident) => {
        impl ReduceMax for $id {
            type Acc = $elem_ty;
            #[cfg(not(target_arch = "aarch64"))]
            #[inline]
            fn reduce_max(self) -> Self::Acc {
                unsafe { $llvm_intr(self) }
            }
            // FIXME: broken on AArch64
            #[cfg(target_arch = "aarch64")]
            #[allow(unused_imports)]
            #[inline]
            fn reduce_max(self) -> Self::Acc {
                use num::Float;
                use cmp::Ord;
                let mut x = self.extract(0);
                for i in 1..$id::lanes() {
                    x = x.max(self.extract(i));
                }
                x
            }
        }
    };
}
red_max!(i8x2, i8, reduce_max_i8x2);
red_max!(u8x2, u8, reduce_max_u8x2);
red_max!(i16x2, i16, reduce_max_i16x2);
red_max!(u16x2, u16, reduce_max_u16x2);
red_max!(i32x2, i32, reduce_max_i32x2);
red_max!(u32x2, u32, reduce_max_u32x2);
red_max!(i64x2, i64, reduce_max_i64x2);
red_max!(u64x2, u64, reduce_max_u64x2);
red_max!(i8x4, i8, reduce_max_i8x4);
red_max!(u8x4, u8, reduce_max_u8x4);
red_max!(i16x4, i16, reduce_max_i16x4);
red_max!(u16x4, u16, reduce_max_u16x4);
red_max!(i32x4, i32, reduce_max_i32x4);
red_max!(u32x4, u32, reduce_max_u32x4);
red_max!(i64x4, i64, reduce_max_i64x4);
red_max!(u64x4, u64, reduce_max_u64x4);
red_max!(i8x8, i8, reduce_max_i8x8);
red_max!(u8x8, u8, reduce_max_u8x8);
red_max!(i16x8, i16, reduce_max_i16x8);
red_max!(u16x8, u16, reduce_max_u16x8);
red_max!(i32x8, i32, reduce_max_i32x8);
red_max!(u32x8, u32, reduce_max_u32x8);
red_max!(i64x8, i64, reduce_max_i64x8);
red_max!(u64x8, u64, reduce_max_u64x8);
red_max!(i8x16, i8, reduce_max_i8x16);
red_max!(u8x16, u8, reduce_max_u8x16);
red_max!(i16x16, i16, reduce_max_i16x16);
red_max!(u16x16, u16, reduce_max_u16x16);
red_max!(i32x16, i32, reduce_max_i32x16);
red_max!(u32x16, u32, reduce_max_u32x16);
red_max!(i8x32, i8, reduce_max_i8x32);
red_max!(u8x32, u8, reduce_max_u8x32);
red_max!(i16x32, i16, reduce_max_i16x32);
red_max!(u16x32, u16, reduce_max_u16x32);
red_max!(i8x64, i8, reduce_max_i8x64);
red_max!(u8x64, u8, reduce_max_u8x64);

red_max!(f32x2, f32, reduce_fmax_f32x2);
red_max!(f64x2, f64, reduce_fmax_f64x2);
red_max!(f32x4, f32, reduce_fmax_f32x4);
red_max!(f64x4, f64, reduce_fmax_f64x4);
red_max!(f32x8, f32, reduce_fmax_f32x8);
red_max!(f64x8, f64, reduce_fmax_f64x8);
red_max!(f32x16, f32, reduce_fmax_f32x16);

#[cfg(test)]
mod tests {
    use super::ReduceMax;
    use ::coresimd::simd::*;

    // note: these are tested in the portable vector API tests

    #[test]
    fn reduce_max_i32x4() {
        let v = i32x4::new(1, 2, -1, 3);
        assert_eq!(v.reduce_max(), 3_i32);
    }
    #[test]
    fn reduce_max_u32x4() {
        let v = u32x4::new(4, 2, 7, 3);
        assert_eq!(v.reduce_max(), 7_u32);
    }
    #[test]
    fn reduce_max_f32x4() {
        let v = f32x4::new(4., 2., -1., 3.);
        assert_eq!(v.reduce_max(), 4.);
    }
}
