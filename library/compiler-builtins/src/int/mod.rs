use core::ops;

macro_rules! hty {
    ($ty:ty) => {
        <$ty as LargeInt>::HighHalf
    }
}

macro_rules! os_ty {
    ($ty:ty) => {
        <$ty as Int>::OtherSign
    }
}

pub mod mul;
pub mod sdiv;
pub mod shift;
pub mod udiv;

/// Trait for some basic operations on integers
pub trait Int:
    Copy +
    PartialEq +
    PartialOrd +
    ops::AddAssign +
    ops::Add<Output = Self> +
    ops::Sub<Output = Self> +
    ops::Div<Output = Self> +
    ops::Shl<u32, Output = Self> +
    ops::Shr<u32, Output = Self> +
    ops::BitOr<Output = Self> +
    ops::BitXor<Output = Self> +
    ops::BitAnd<Output = Self> +
    ops::BitAndAssign +
    ops::Not<Output = Self> +
{
    /// Type with the same width but other signedness
    type OtherSign: Int;
    /// Unsigned version of Self
    type UnsignedInt: Int;

    /// Returns the bitwidth of the int type
    fn bits() -> u32;

    fn zero() -> Self;
    fn one() -> Self;

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

    // copied from primitive integers, but put in a trait
    fn max_value() -> Self;
    fn min_value() -> Self;
    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn aborting_div(self, other: Self) -> Self;
    fn aborting_rem(self, other: Self) -> Self;
}

fn unwrap<T>(t: Option<T>) -> T {
    match t {
        Some(t) => t,
        None => ::abort(),
    }
}

macro_rules! int_impl {
    ($ity:ty, $uty:ty, $bits:expr) => {
        impl Int for $uty {
            type OtherSign = $ity;
            type UnsignedInt = $uty;

            fn zero() -> Self {
                0
            }

            fn one() -> Self {
                1
            }

            fn bits() -> u32 {
                $bits
            }

            fn extract_sign(self) -> (bool, $uty) {
                (false, self)
            }

            fn unsigned(self) -> $uty {
                self
            }

            fn from_unsigned(me: $uty) -> Self {
                me
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

            fn aborting_div(self, other: Self) -> Self {
                unwrap(<Self>::checked_div(self, other))
            }

            fn aborting_rem(self, other: Self) -> Self {
                unwrap(<Self>::checked_rem(self, other))
            }
        }

        impl Int for $ity {
            type OtherSign = $uty;
            type UnsignedInt = $uty;

            fn bits() -> u32 {
                $bits
            }

            fn zero() -> Self {
                0
            }

            fn one() -> Self {
                1
            }

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

            fn aborting_div(self, other: Self) -> Self {
                unwrap(<Self>::checked_div(self, other))
            }

            fn aborting_rem(self, other: Self) -> Self {
                unwrap(<Self>::checked_rem(self, other))
            }
        }
    }
}

int_impl!(i32, u32, 32);
int_impl!(i64, u64, 64);
int_impl!(i128, u128, 128);

/// Trait to convert an integer to/from smaller parts
pub trait LargeInt: Int {
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
    }
}

large_int!(u64, u32, u32, 32);
large_int!(i64, u32, i32, 32);
large_int!(u128, u64, u64, 64);
large_int!(i128, u64, i64, 64);
