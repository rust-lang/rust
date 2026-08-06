//! Various traits used to restrict intrinsics to not-completely-wrong types.

use crate::marker::PointeeSized;

/// Types with a built-in dereference operator in runtime MIR,
/// aka references and raw pointers.
///
/// # Safety
/// Must actually *be* such a type.
pub unsafe trait BuiltinDeref: Sized {
    type Pointee: PointeeSized;
}

unsafe impl<T: PointeeSized> BuiltinDeref for &mut T {
    type Pointee = T;
}
unsafe impl<T: PointeeSized> BuiltinDeref for &T {
    type Pointee = T;
}
unsafe impl<T: PointeeSized> BuiltinDeref for *mut T {
    type Pointee = T;
}
unsafe impl<T: PointeeSized> BuiltinDeref for *const T {
    type Pointee = T;
}

pub trait ChangePointee<U: PointeeSized>: BuiltinDeref {
    type Output;
}
impl<'a, T: PointeeSized + 'a, U: PointeeSized + 'a> ChangePointee<U> for &'a mut T {
    type Output = &'a mut U;
}
impl<'a, T: PointeeSized + 'a, U: PointeeSized + 'a> ChangePointee<U> for &'a T {
    type Output = &'a U;
}
impl<T: PointeeSized, U: PointeeSized> ChangePointee<U> for *mut T {
    type Output = *mut U;
}
impl<T: PointeeSized, U: PointeeSized> ChangePointee<U> for *const T {
    type Output = *const U;
}

/// Built-in float types (f16, f32, f64 and f128).
///
/// # Safety
/// Must actually *be* such a type.
#[rustc_const_unstable(feature = "core_intrinsics", issue = "none")]
pub const unsafe trait FloatPrimitive: Sized + Copy {
    type UInt: const core::ops::BitOr<Output = Self::UInt>
        + const core::ops::BitAnd<Output = Self::UInt>
        + const core::ops::Not<Output = Self::UInt>;
    const SIGN_MASK: Self::UInt;
    fn to_bits(self) -> Self::UInt;
    fn from_bits(bits: Self::UInt) -> Self;
}

#[rustc_const_unstable(feature = "core_intrinsics", issue = "none")]
const unsafe impl FloatPrimitive for f16 {
    type UInt = u16;
    const SIGN_MASK: Self::UInt = f16::SIGN_MASK;
    #[inline]
    fn to_bits(self) -> Self::UInt {
        f16::to_bits(self)
    }
    #[inline]
    fn from_bits(bits: Self::UInt) -> Self {
        f16::from_bits(bits)
    }
}

#[rustc_const_unstable(feature = "core_intrinsics", issue = "none")]
const unsafe impl FloatPrimitive for f32 {
    type UInt = u32;
    const SIGN_MASK: Self::UInt = f32::SIGN_MASK;
    #[inline]
    fn to_bits(self) -> Self::UInt {
        f32::to_bits(self)
    }
    #[inline]
    fn from_bits(bits: Self::UInt) -> Self {
        f32::from_bits(bits)
    }
}

#[rustc_const_unstable(feature = "core_intrinsics", issue = "none")]
const unsafe impl FloatPrimitive for f64 {
    type UInt = u64;
    const SIGN_MASK: Self::UInt = f64::SIGN_MASK;
    #[inline]
    fn to_bits(self) -> Self::UInt {
        f64::to_bits(self)
    }
    #[inline]
    fn from_bits(bits: Self::UInt) -> Self {
        f64::from_bits(bits)
    }
}

#[rustc_const_unstable(feature = "core_intrinsics", issue = "none")]
const unsafe impl FloatPrimitive for f128 {
    type UInt = u128;
    const SIGN_MASK: Self::UInt = f128::SIGN_MASK;
    #[inline]
    fn to_bits(self) -> Self::UInt {
        f128::to_bits(self)
    }
    #[inline]
    fn from_bits(bits: Self::UInt) -> Self {
        f128::from_bits(bits)
    }
}
