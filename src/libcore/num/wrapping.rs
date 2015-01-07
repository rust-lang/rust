#![allow(missing_docs)]

use ops::*;

#[cfg(not(stage0))]
use intrinsics::{overflowing_add, overflowing_sub, overflowing_mul};

pub trait WrappingOps {
    fn wrapping_add(self, rhs: Self) -> Self;
    fn wrapping_sub(self, rhs: Self) -> Self;
    fn wrapping_mul(self, rhs: Self) -> Self;
}

#[cfg(not(stage0))]
macro_rules! wrapping_impl {
    ($($t:ty)*) => ($(
        impl WrappingOps for $t {
            #[inline(always)]
            fn wrapping_add(self, rhs: $t) -> $t {
                unsafe {
                    overflowing_add(self, rhs)
                }
            }
            #[inline(always)]
            fn wrapping_sub(self, rhs: $t) -> $t {
                unsafe {
                    overflowing_sub(self, rhs)
                }
            }
            #[inline(always)]
            fn wrapping_mul(self, rhs: $t) -> $t {
                unsafe {
                    overflowing_mul(self, rhs)
                }
            }
        }
    )*)
}

#[cfg(stage0)]
macro_rules! wrapping_impl {
    ($($t:ty)*) => ($(
        impl WrappingOps for $t {
            #[inline(always)]
            fn wrapping_add(self, rhs: $t) -> $t {
                self + rhs
            }
            #[inline(always)]
            fn wrapping_sub(self, rhs: $t) -> $t {
                self - rhs
            }
            #[inline(always)]
            fn wrapping_mul(self, rhs: $t) -> $t {
                self * rhs
            }
        }
    )*)
}

wrapping_impl! { uint u8 u16 u32 u64 int i8 i16 i32 i64 }

#[derive(PartialEq,Eq,PartialOrd,Ord,Clone)]
pub struct Wrapping<T>(pub T);

impl<T:WrappingOps> Add for Wrapping<T> {
    type Output = Wrapping<T>;

    #[inline(always)]
    fn add(self, other: Wrapping<T>) -> Wrapping<T> {
        Wrapping(self.0.wrapping_add(other.0))
    }
}

impl<T:WrappingOps> Sub for Wrapping<T> {
    type Output = Wrapping<T>;

    #[inline(always)]
    fn sub(self, other: Wrapping<T>) -> Wrapping<T> {
        Wrapping(self.0.wrapping_sub(other.0))
    }
}

impl<T:WrappingOps> Mul for Wrapping<T> {
    type Output = Wrapping<T>;

    #[inline(always)]
    fn mul(self, other: Wrapping<T>) -> Wrapping<T> {
        Wrapping(self.0.wrapping_mul(other.0))
    }
}
