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
pub unsafe trait FloatPrimitive:
    Sized + Copy + PartialOrd + core::ops::Add<Output = Self>
{
    type UInt: core::ops::BitOr<Output = Self::UInt>
        + core::ops::BitAnd<Output = Self::UInt>
        + core::ops::Not<Output = Self::UInt>;
    const SIGN_MASK: Self::UInt;
    fn to_bits(self) -> Self::UInt;
    fn from_bits(bits: Self::UInt) -> Self;
    fn is_nan(self) -> bool;
    fn is_sign_positive(self) -> bool;
    fn is_sign_negative(self) -> bool;
}

macro_rules! impl_float_primitive {
    ($($float:ident => $bits:ident),+) => {$(
        unsafe impl FloatPrimitive for $float {
            type UInt = $bits;
            const SIGN_MASK: Self::UInt = $float::SIGN_MASK;
            #[inline]
            fn to_bits(self) -> Self::UInt {
                $float::to_bits(self)
            }
            #[inline]
            fn from_bits(bits: Self::UInt) -> Self {
                $float::from_bits(bits)
            }
            #[inline]
            fn is_nan(self) -> bool {
                $float::is_nan(self)
            }
            #[inline]
            fn is_sign_positive(self) -> bool {
                $float::is_sign_positive(self)
            }
            #[inline]
            fn is_sign_negative(self) -> bool {
                $float::is_sign_negative(self)
            }
        }
    )+};
}
impl_float_primitive!(f16 => u16, f32 => u32, f64 => u64, f128 => u128);

/// Built-in integer types (i8, i16, .., i128, isize, u8, u16, .., u128, usize).
///
/// Intentionally does not include other integer-repr types like `bool` or `char`.
///
/// # Safety
/// Must actually *be* such a type.
#[rustc_const_unstable(feature = "core_intrinsics", issue = "none")]
pub const unsafe trait IntegerPrimitive: Copy + [const] Ord {}

macro_rules! impl_integer_primitive {
    ($($t:ty),*) => {$(
        #[rustc_const_unstable(feature = "core_intrinsics", issue = "none")]
        const unsafe impl IntegerPrimitive for $t {}
    )*};
}
impl_integer_primitive!(i8, i16, i32, i64, i128, isize);
impl_integer_primitive!(u8, u16, u32, u64, u128, usize);
