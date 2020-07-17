use core::ops;

pub mod addsub;
pub mod leading_zeros;
pub mod mul;
pub mod sdiv;
pub mod shift;
pub mod udiv;

pub use self::leading_zeros::__clzsi2;

/// Trait for some basic operations on integers
pub(crate) trait Int:
    Copy
    + PartialEq
    + PartialOrd
    + ops::AddAssign
    + ops::BitAndAssign
    + ops::BitOrAssign
    + ops::ShlAssign<i32>
    + ops::ShrAssign<u32>
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Div<Output = Self>
    + ops::Shl<u32, Output = Self>
    + ops::Shr<u32, Output = Self>
    + ops::BitOr<Output = Self>
    + ops::BitXor<Output = Self>
    + ops::BitAnd<Output = Self>
    + ops::Not<Output = Self>
{
    /// Type with the same width but other signedness
    type OtherSign: Int;
    /// Unsigned version of Self
    type UnsignedInt: Int;

    /// The bitwidth of the int type
    const BITS: u32;

    const ZERO: Self;
    const ONE: Self;

    /// Extracts the sign from self and returns a tuple.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let i = -25_i32;
    /// let (sign, u) = i.extract_sign();
    /// assert_eq!(sign, true);
    /// assert_eq!(u, 25_u32);
    /// ```
    fn extract_sign(self) -> (bool, Self::UnsignedInt);

    fn unsigned(self) -> Self::UnsignedInt;
    fn from_unsigned(unsigned: Self::UnsignedInt) -> Self;

    fn from_bool(b: bool) -> Self;

    // copied from primitive integers, but put in a trait
    fn max_value() -> Self;
    fn min_value() -> Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_shl(self, other: u32) -> Self;
    fn overflowing_add(self, other: Self) -> (Self, bool);
    fn aborting_div(self, other: Self) -> Self;
    fn aborting_rem(self, other: Self) -> Self;
    fn leading_zeros(self) -> u32;
}

fn unwrap<T>(t: Option<T>) -> T {
    match t {
        Some(t) => t,
        None => ::abort(),
    }
}

macro_rules! int_impl_common {
    ($ty:ty, $bits:expr) => {
        const BITS: u32 = $bits;

        const ZERO: Self = 0;
        const ONE: Self = 1;

        fn from_bool(b: bool) -> Self {
            b as $ty
        }

        fn max_value() -> Self {
            <Self>::max_value()
        }

        fn min_value() -> Self {
            <Self>::min_value()
        }

        fn wrapping_add(self, other: Self) -> Self {
            <Self>::wrapping_add(self, other)
        }

        fn wrapping_mul(self, other: Self) -> Self {
            <Self>::wrapping_mul(self, other)
        }

        fn wrapping_sub(self, other: Self) -> Self {
            <Self>::wrapping_sub(self, other)
        }

        fn wrapping_shl(self, other: u32) -> Self {
            <Self>::wrapping_shl(self, other)
        }

        fn overflowing_add(self, other: Self) -> (Self, bool) {
            <Self>::overflowing_add(self, other)
        }

        fn aborting_div(self, other: Self) -> Self {
            unwrap(<Self>::checked_div(self, other))
        }

        fn aborting_rem(self, other: Self) -> Self {
            unwrap(<Self>::checked_rem(self, other))
        }

        fn leading_zeros(self) -> u32 {
            <Self>::leading_zeros(self)
        }
    };
}

macro_rules! int_impl {
    ($ity:ty, $uty:ty, $bits:expr) => {
        impl Int for $uty {
            type OtherSign = $ity;
            type UnsignedInt = $uty;

            fn extract_sign(self) -> (bool, $uty) {
                (false, self)
            }

            fn unsigned(self) -> $uty {
                self
            }

            fn from_unsigned(me: $uty) -> Self {
                me
            }

            int_impl_common!($uty, $bits);
        }

        impl Int for $ity {
            type OtherSign = $uty;
            type UnsignedInt = $uty;

            fn extract_sign(self) -> (bool, $uty) {
                if self < 0 {
                    (true, (!(self as $uty)).wrapping_add(1))
                } else {
                    (false, self as $uty)
                }
            }

            fn unsigned(self) -> $uty {
                self as $uty
            }

            fn from_unsigned(me: $uty) -> Self {
                me as $ity
            }

            int_impl_common!($ity, $bits);
        }
    };
}

int_impl!(i32, u32, 32);
int_impl!(i64, u64, 64);
int_impl!(i128, u128, 128);

/// Trait to convert an integer to/from smaller parts
pub(crate) trait LargeInt: Int {
    type LowHalf: Int;
    type HighHalf: Int;

    fn low(self) -> Self::LowHalf;
    fn low_as_high(low: Self::LowHalf) -> Self::HighHalf;
    fn high(self) -> Self::HighHalf;
    fn high_as_low(low: Self::HighHalf) -> Self::LowHalf;
    fn from_parts(low: Self::LowHalf, high: Self::HighHalf) -> Self;
}

macro_rules! large_int {
    ($ty:ty, $tylow:ty, $tyhigh:ty, $halfbits:expr) => {
        impl LargeInt for $ty {
            type LowHalf = $tylow;
            type HighHalf = $tyhigh;

            fn low(self) -> $tylow {
                self as $tylow
            }
            fn low_as_high(low: $tylow) -> $tyhigh {
                low as $tyhigh
            }
            fn high(self) -> $tyhigh {
                (self >> $halfbits) as $tyhigh
            }
            fn high_as_low(high: $tyhigh) -> $tylow {
                high as $tylow
            }
            fn from_parts(low: $tylow, high: $tyhigh) -> $ty {
                low as $ty | ((high as $ty) << $halfbits)
            }
        }
    };
}

large_int!(u64, u32, u32, 32);
large_int!(i64, u32, i32, 32);
large_int!(u128, u64, u64, 64);
large_int!(i128, u64, i64, 64);

/// Trait to express (possibly lossy) casting of integers
pub(crate) trait CastInto<T: Copy>: Copy {
    fn cast(self) -> T;
}

macro_rules! cast_into {
    ($ty:ty) => {
        cast_into!($ty; usize, isize, u32, i32, u64, i64, u128, i128);
    };
    ($ty:ty; $($into:ty),*) => {$(
        impl CastInto<$into> for $ty {
            fn cast(self) -> $into {
                self as $into
            }
        }
    )*};
}

cast_into!(u32);
cast_into!(i32);
cast_into!(u64);
cast_into!(i64);
cast_into!(u128);
cast_into!(i128);

pub(crate) trait WideInt: Int {
    type Output: Int;

    fn wide_mul(self, other: Self) -> (Self, Self);
    fn wide_shift_left(&mut self, low: &mut Self, count: i32);
    fn wide_shift_right_with_sticky(&mut self, low: &mut Self, count: i32);
}

macro_rules! impl_wide_int {
    ($ty:ty, $tywide:ty, $bits:expr) => {
        impl WideInt for $ty {
            type Output = $ty;

            fn wide_mul(self, other: Self) -> (Self, Self) {
                let product = (self as $tywide).wrapping_mul(other as $tywide);
                ((product >> ($bits as $ty)) as $ty, product as $ty)
            }

            fn wide_shift_left(&mut self, low: &mut Self, count: i32) {
                *self = (*self << count) | (*low >> ($bits - count));
                *low = *low << count;
            }

            fn wide_shift_right_with_sticky(&mut self, low: &mut Self, count: i32) {
                if count < $bits {
                    let sticky = *low << ($bits - count);
                    *low = *self << ($bits - count) | *low >> count | sticky;
                    *self = *self >> count;
                } else if count < 2 * $bits {
                    let sticky = *self << (2 * $bits - count) | *low;
                    *low = *self >> (count - $bits) | sticky;
                    *self = 0;
                } else {
                    let sticky = *self | *low;
                    *self = sticky;
                    *self = 0;
                }
            }
        }
    };
}

impl_wide_int!(u32, u64, 32);
impl_wide_int!(u64, u128, 64);
