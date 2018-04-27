#![feature(stdsimd, sse4a_target_feature, avx512_target_feature)]
#![feature(arm_target_feature)]
#![feature(aarch64_target_feature)]
#![feature(powerpc_target_feature)]
#![allow(unused_attributes)]

#[macro_use]
extern crate stdsimd;

use stdsimd::simd::*;

#[cfg(target_arch = "powerpc")]
macro_rules! is_powerpc_feature_detected {
    ($t:tt) => {
        false
    };
}

macro_rules! invoke_arch {
    ($macro:ident, $feature_macro:ident, $id:ident, $elem_ty:ident,
     [$($feature:tt),*]) => {
        $($macro!($feature, $feature_macro, $id, $elem_ty);)*
    }
}

macro_rules! invoke_vectors {
    ($macro:ident, [$(($id:ident, $elem_ty:ident)),*]) => {
        $(
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            invoke_arch!($macro, is_x86_feature_detected, $id, $elem_ty,
                        ["sse", "sse2", "sse3", "ssse3", "sse4.1",
                         "sse4.2", "sse4a", "avx2", "avx2", "avx512f"]);
            #[cfg(target_arch = "aarch64")]
            invoke_arch!($macro, is_aarch64_feature_detected, $id, $elem_ty,
                        ["neon"]);
            #[cfg(all(target_arch = "arm", target_feature = "v7", target_feature = "neon"))]
            invoke_arch!($macro, is_arm_feature_detected, $id, $elem_ty,
                         ["neon"]);
            #[cfg(target_arch = "powerpc")]
            invoke_arch!($macro, is_powerpc_feature_detected, $id, $elem_ty, ["altivec"]);
            #[cfg(target_arch = "powerpc64")]
            invoke_arch!($macro, is_powerpc64_feature_detected, $id, $elem_ty, ["altivec"]);
        )*
    }
}

macro_rules! finvoke {
    ($macro:ident) => {
        invoke_vectors!(
            $macro,
            [
                (f32x2, f32),
                (f32x4, f32),
                (f32x8, f32),
                (f32x16, f32),
                (f64x2, f64),
                (f64x4, f64),
                (f64x8, f64)
            ]
        );
    };
}

macro_rules! iinvoke {
    ($macro:ident) => {
        invoke_vectors!(
            $macro,
            [
                (i8x2, i8),
                (i8x4, i8),
                (i8x8, i8),
                (i8x16, i8),
                (i8x32, i8),
                (i8x64, i8),
                (i16x2, i16),
                (i16x4, i16),
                (i16x8, i16),
                (i16x16, i16),
                (i16x32, i16),
                (i32x2, i32),
                (i32x4, i32),
                (i32x8, i32),
                (i32x16, i32),
                (i64x2, i64),
                (i64x4, i64),
                (i64x8, i64),
                (u8x2, u8),
                (u8x4, u8),
                (u8x8, u8),
                (u8x16, u8),
                (u8x32, u8),
                (u8x64, u8),
                (u16x2, u16),
                (u16x4, u16),
                (u16x8, u16),
                (u16x16, u16),
                (u16x32, u16),
                (u32x2, u32),
                (u32x4, u32),
                (u32x8, u32),
                (u32x16, u32),
                (u64x2, u64),
                (u64x4, u64),
                (u64x8, u64)
            ]
        );
    };
}

macro_rules! min_nan_test {
    ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
        if $feature_macro!($feature) {
            #[target_feature(enable = $feature)]
            unsafe fn test_fn() {
                let n0 = ::std::$elem_ty::NAN;

                assert_eq!(n0.min(-3.0), -3.0);
                assert_eq!((-3.0 as $elem_ty).min(n0), -3.0);

                let v0 = $id::splat(-3.0);

                // FIXME (https://github.com/rust-lang-nursery/stdsimd/issues/408):
                // When the last element is NaN the current implementation produces incorrect results.
                let bugbug = 1;
                for i in 0..$id::lanes() - bugbug {
                    let mut v = v0.replace(i, n0);
                    // If there is a NaN, the result is always the smallest element:
                    assert_eq!(v.min_element(), -3.0, "nan at {} => {} | {:?} | {:X}", i, v.min_element(), v, v.as_int());
                    for j in 0..i {
                        v = v.replace(j, n0);
                        assert_eq!(v.min_element(), -3.0, "nan at {} => {} | {:?} | {:X}", i, v.min_element(), v, v.as_int());
                    }
                }
                // If the vector contains all NaNs the result is NaN:
                let vn = $id::splat(n0);
                assert!(vn.min_element().is_nan(), "all nans | v={:?} | min={} | is_nan: {}",
                        vn, vn.min_element(), vn.min_element().is_nan());
            }
            unsafe { test_fn() };
        }
    }
}

#[test]
fn min_nan() {
    finvoke!(min_nan_test);
}

macro_rules! max_nan_test {
    ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
        if $feature_macro!($feature) {
            #[target_feature(enable = $feature)]
            unsafe fn test_fn() {
                let n0 = ::std::$elem_ty::NAN;

                assert_eq!(n0.max(-3.0), -3.0);
                assert_eq!((-3.0 as $elem_ty).max(n0), -3.0);

                let v0 = $id::splat(-3.0);

                // FIXME (https://github.com/rust-lang-nursery/stdsimd/issues/408):
                // When the last element is NaN the current implementation produces incorrect results.
                let bugbug = 1;
                for i in 0..$id::lanes() - bugbug {
                    let mut v = v0.replace(i, n0);
                    // If there is a NaN the result is always the largest element:
                    assert_eq!(v.max_element(), -3.0, "nan at {} => {} | {:?} | {:X}", i, v.max_element(), v, v.as_int());
                    for j in 0..i {
                        v = v.replace(j, n0);
                        assert_eq!(v.max_element(), -3.0, "nan at {} => {} | {:?} | {:X}", i, v.max_element(), v, v.as_int());
                    }
                }

                // If the vector contains all NaNs the result is NaN:
                let vn = $id::splat(n0);
                assert!(vn.max_element().is_nan(), "all nans | v={:?} | max={} | is_nan: {}",
                        vn, vn.max_element(), vn.max_element().is_nan());
            }
            unsafe { test_fn() };
        }
    }
}

#[test]
fn max_nan() {
    finvoke!(max_nan_test);
}

macro_rules! wrapping_sum_nan_test {
    ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
        if $feature_macro!($feature) {
            #[target_feature(enable = $feature)]
            #[allow(unreachable_code)]
            unsafe fn test_fn() {
                // FIXME: https://bugs.llvm.org/show_bug.cgi?id=36732
                // https://github.com/rust-lang-nursery/stdsimd/issues/409
                return;

                let n0 = ::std::$elem_ty::NAN;
                let v0 = $id::splat(-3.0);
                for i in 0..$id::lanes() {
                    let mut v = v0.replace(i, n0);
                    // If the vector contains a NaN the result is NaN:
                    assert!(
                        v.wrapping_sum().is_nan(),
                        "nan at {} => {} | {:?}",
                        i,
                        v.wrapping_sum(),
                        v
                    );
                    for j in 0..i {
                        v = v.replace(j, n0);
                        assert!(v.wrapping_sum().is_nan());
                    }
                }
                let v = $id::splat(n0);
                assert!(v.wrapping_sum().is_nan(), "all nans | {:?}", v);
            }
            unsafe { test_fn() };
        }
    };
}

#[test]
fn wrapping_sum_nan() {
    finvoke!(wrapping_sum_nan_test);
}

macro_rules! wrapping_product_nan_test {
    ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
        if $feature_macro!($feature) {
            #[target_feature(enable = $feature)]
            #[allow(unreachable_code)]
            unsafe fn test_fn() {
                // FIXME: https://bugs.llvm.org/show_bug.cgi?id=36732
                // https://github.com/rust-lang-nursery/stdsimd/issues/409
                return;

                let n0 = ::std::$elem_ty::NAN;
                let v0 = $id::splat(-3.0);
                for i in 0..$id::lanes() {
                    let mut v = v0.replace(i, n0);
                    // If the vector contains a NaN the result is NaN:
                    assert!(
                        v.wrapping_product().is_nan(),
                        "nan at {} | {:?}",
                        i,
                        v
                    );
                    for j in 0..i {
                        v = v.replace(j, n0);
                        assert!(v.wrapping_sum().is_nan());
                    }
                }
                let v = $id::splat(n0);
                assert!(
                    v.wrapping_product().is_nan(),
                    "all nans | {:?}",
                    v
                );
            }
            unsafe { test_fn() };
        }
    };
}

#[test]
fn wrapping_product_nan() {
    finvoke!(wrapping_product_nan_test);
}

trait AsInt {
    type Int;
    fn as_int(self) -> Self::Int;
    fn from_int(Self::Int) -> Self;
}

macro_rules! as_int {
    ($float:ident, $int:ident) => {
        impl AsInt for $float {
            type Int = $int;
            fn as_int(self) -> $int {
                unsafe { ::std::mem::transmute(self) }
            }
            fn from_int(x: $int) -> $float {
                unsafe { ::std::mem::transmute(x) }
            }
        }
    };
}

as_int!(f32, u32);
as_int!(f64, u64);
as_int!(f32x2, i32x2);
as_int!(f32x4, i32x4);
as_int!(f32x8, i32x8);
as_int!(f32x16, i32x16);
as_int!(f64x2, i64x2);
as_int!(f64x4, i64x4);
as_int!(f64x8, i64x8);

// FIXME: these fail on i586 for some reason
#[cfg(not(all(target_arch = "x86", not(target_feature = "sse2"))))]
mod offset {
    use super::*;

    trait TreeReduceAdd {
        type R;
        fn tree_reduce_add(self) -> Self::R;
    }

    macro_rules! tree_reduce_add_f {
    ($elem_ty:ident) => {
        impl<'a> TreeReduceAdd for &'a [$elem_ty] {
            type R = $elem_ty;
            fn tree_reduce_add(self) -> $elem_ty {
                if self.len() == 2 {
                    println!("  lv: {}, rv: {} => {}", self[0], self[1], self[0] + self[1]);
                    self[0] + self[1]
                } else {
                    let mid = self.len() / 2;
                    let (left, right) = self.split_at(mid);
                    println!("  splitting self: {:?} at mid {} into left: {:?}, right: {:?}", self, mid, self[0], self[1]);
                    Self::tree_reduce_add(left) + Self::tree_reduce_add(right)
                }
            }
        }
    };
}
    tree_reduce_add_f!(f32);
    tree_reduce_add_f!(f64);

    macro_rules! wrapping_sum_roundoff_test {
    ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
        if $feature_macro!($feature) {
            #[target_feature(enable = $feature)]
            unsafe fn test_fn() {
                let mut start = std::$elem_ty::EPSILON;
                let mut wrapping_sum = 0. as $elem_ty;

                let mut v = $id::splat(0. as $elem_ty);
                for i in 0..$id::lanes() {
                    let c = if i % 2 == 0 { 1e3 } else { -1. };
                    start *= 3.14 * c;
                    wrapping_sum += start;
                    // println!("{} | start: {}", stringify!($id), start);
                    v = v.replace(i, start);
                }
                let vwrapping_sum = v.wrapping_sum();
                println!(
                    "{} | lwrapping_sum: {}",
                    stringify!($id),
                    wrapping_sum
                );
                println!(
                    "{} | vwrapping_sum: {}",
                    stringify!($id),
                    vwrapping_sum
                );
                let r = vwrapping_sum.as_int() == wrapping_sum.as_int();
                // This is false in general; the intrinsic performs a
                // tree-reduce:
                println!("{} | equal: {}", stringify!($id), r);

                let mut a = [0. as $elem_ty; $id::lanes()];
                v.store_unaligned(&mut a);

                let twrapping_sum = a.tree_reduce_add();
                println!(
                    "{} | twrapping_sum: {}",
                    stringify!($id),
                    twrapping_sum
                );

                // tolerate 1 ULP difference:
                if vwrapping_sum.as_int() > twrapping_sum.as_int() {
                    assert!(
                        vwrapping_sum.as_int() - twrapping_sum.as_int()
                            < 2,
                        "v: {:?} | vwrapping_sum: {} | twrapping_sum: {}",
                        v,
                        vwrapping_sum,
                        twrapping_sum
                    );
                } else {
                    assert!(
                        twrapping_sum.as_int() - vwrapping_sum.as_int()
                            < 2,
                        "v: {:?} | vwrapping_sum: {} | twrapping_sum: {}",
                        v,
                        vwrapping_sum,
                        twrapping_sum
                    );
                }
            }
            unsafe { test_fn() };
        }
    };
}

    #[test]
    fn wrapping_sum_roundoff_test() {
        finvoke!(wrapping_sum_roundoff_test);
    }

    trait TreeReduceMul {
        type R;
        fn tree_reduce_mul(self) -> Self::R;
    }

    macro_rules! tree_reduce_mul_f {
    ($elem_ty:ident) => {
        impl<'a> TreeReduceMul for &'a [$elem_ty] {
            type R = $elem_ty;
            fn tree_reduce_mul(self) -> $elem_ty {
                if self.len() == 2 {
                    println!("  lv: {}, rv: {} => {}", self[0], self[1], self[0] * self[1]);
                    self[0] * self[1]
                } else {
                    let mid = self.len() / 2;
                    let (left, right) = self.split_at(mid);
                    println!("  splitting self: {:?} at mid {} into left: {:?}, right: {:?}", self, mid, self[0], self[1]);
                    Self::tree_reduce_mul(left) * Self::tree_reduce_mul(right)
                }
            }
        }
    };
}

    tree_reduce_mul_f!(f32);
    tree_reduce_mul_f!(f64);

    macro_rules! wrapping_product_roundoff_test {
        ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
            if $feature_macro!($feature) {
                #[target_feature(enable = $feature)]
                unsafe fn test_fn() {
                    let mut start = std::$elem_ty::EPSILON;
                    let mut mul = 1. as $elem_ty;

                    let mut v = $id::splat(1. as $elem_ty);
                    for i in 0..$id::lanes() {
                        let c = if i % 2 == 0 { 1e3 } else { -1. };
                        start *= 3.14 * c;
                        mul *= start;
                        println!("{} | start: {}", stringify!($id), start);
                        v = v.replace(i, start);
                    }
                    let vmul = v.wrapping_product();
                    println!("{} | lmul: {}", stringify!($id), mul);
                    println!("{} | vmul: {}", stringify!($id), vmul);
                    let r = vmul.as_int() == mul.as_int();
                    // This is false in general; the intrinsic performs a
                    // tree-reduce:
                    println!("{} | equal: {}", stringify!($id), r);

                    let mut a = [0. as $elem_ty; $id::lanes()];
                    v.store_unaligned(&mut a);

                    let tmul = a.tree_reduce_mul();
                    println!("{} | tmul: {}", stringify!($id), tmul);

                    // tolerate 1 ULP difference:
                    if vmul.as_int() > tmul.as_int() {
                        assert!(
                            vmul.as_int() - tmul.as_int() < 2,
                            "v: {:?} | vmul: {} | tmul: {}",
                            v,
                            vmul,
                            tmul
                        );
                    } else {
                        assert!(
                            tmul.as_int() - vmul.as_int() < 2,
                            "v: {:?} | vmul: {} | tmul: {}",
                            v,
                            vmul,
                            tmul
                        );
                    }
                }
                unsafe { test_fn() };
            }
        };
    }

    #[test]
    fn wrapping_product_roundoff_test() {
        finvoke!(wrapping_product_roundoff_test);
    }

    macro_rules! wrapping_sum_overflow_test {
        ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
            if $feature_macro!($feature) {
                #[target_feature(enable = $feature)]
                unsafe fn test_fn() {
                    let start = $elem_ty::max_value()
                        - ($id::lanes() as $elem_ty / 2);

                    let v = $id::splat(start as $elem_ty);
                    let vwrapping_sum = v.wrapping_sum();

                    let mut wrapping_sum = start;
                    for _ in 1..$id::lanes() {
                        wrapping_sum = wrapping_sum.wrapping_add(start);
                    }
                    assert_eq!(wrapping_sum, vwrapping_sum, "v = {:?}", v);
                }
                unsafe { test_fn() };
            }
        };
    }

    #[test]
    fn wrapping_sum_overflow_test() {
        iinvoke!(wrapping_sum_overflow_test);
    }

    macro_rules! mul_overflow_test {
        ($feature:tt, $feature_macro:ident, $id:ident, $elem_ty:ident) => {
            if $feature_macro!($feature) {
                #[target_feature(enable = $feature)]
                unsafe fn test_fn() {
                    let start = $elem_ty::max_value()
                        - ($id::lanes() as $elem_ty / 2);

                    let v = $id::splat(start as $elem_ty);
                    let vmul = v.wrapping_product();

                    let mut mul = start;
                    for _ in 1..$id::lanes() {
                        mul = mul.wrapping_mul(start);
                    }
                    assert_eq!(mul, vmul, "v = {:?}", v);
                }
                unsafe { test_fn() };
            }
        };
    }

    #[test]
    fn mul_overflow_test() {
        iinvoke!(mul_overflow_test);
    }

}
