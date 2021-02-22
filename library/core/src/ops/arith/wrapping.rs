use crate::sealed::Sealed;

/// The trait corresponds to wrapping add arithmetics.
///
/// # Notes
///
/// This trait is sealed, you cannot implement this trait outside the standard library.
/// This trait doesn't correspond to any operator.
#[unstable(feature = "wrapping_add", issue = "none")]
pub trait WrappingAdd<Rhs = Self>: Sealed + Copy {
    /// The resulting type after applying the `wrapping_add` operator.
    type Output;

    /// Performs the add and wrapping operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::WrappingAdd;
    /// assert_eq!(<u16 as WrappingAdd>::wrapping_add(100, 27), 127);
    /// assert_eq!(<u16 as WrappingAdd>::wrapping_add(u16::MAX, 2), 1);
    /// ```
    #[must_use]
    fn wrapping_add(self, rhs: Rhs) -> Self::Output;
}

macro_rules! add_impl {
    ($($T:ty)+) => {
        $(
            #[unstable(feature = "wrapping_add", issue = "none")]
            impl WrappingAdd for $T {
                type Output = $T;

                #[inline]
                fn wrapping_add(self, other: $T) -> $T {
                    <$T>::wrapping_add(self, other)
                }
            }

            forward_ref_binop! { impl WrappingAdd, wrapping_add for $T, $T }
        )+
    };
}

add_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

/// The trait for assignment result of `wrapping_add` operation.
//
/// # Notes
///
/// This trait is sealed, you cannot implement this trait outside the standard library.
/// This trait doesn't correspond to any operator.
#[unstable(feature = "wrapping_add", issue = "none")]
pub trait WrappingAddAssign<Rhs = Self>: Sealed + Copy {
    /// Assign result of `wrapping_add` to `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::WrappingAddAssign;
    /// let mut x = 42u16;
    /// assert_eq!(x.wrapping_add_assign(1), 43);
    /// assert_eq!(x.wrapping_add_assign(u16::MAX), 42);
    /// ```
    fn wrapping_add_assign(&mut self, rhs: Rhs);
}

macro_rules! add_assign_impl {
    ($($T:ty)+) => {
        $(
            #[unstable(feature = "wrapping_add", issue = "none")]
            impl WrappingAddAssign for $T {
                #[inline]
                fn wrapping_add_assign(&mut self, other: $T) {
                    *self = <$T>::wrapping_add(*self, other);
                }
            }

            forward_ref_op_assign! { impl WrappingAddAssign, wrapping_add_assign for $T, $T }
        )+
    };
}

add_assign_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }
