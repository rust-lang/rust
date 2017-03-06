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
pub trait Int {
    /// Type with the same width but other signedness
    type OtherSign;
    /// Unsigned version of Self
    type UnsignedInt;

    /// Returns the bitwidth of the int type
    fn bits() -> u32;

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
}

macro_rules! int_impl {
    ($ity:ty, $uty:ty, $bits:expr) => {
        impl Int for $uty {
            type OtherSign = $ity;
            type UnsignedInt = $uty;

            fn bits() -> u32 {
                $bits
            }

            fn extract_sign(self) -> (bool, $uty) {
                (false, self)
            }
        }

        impl Int for $ity {
            type OtherSign = $uty;
            type UnsignedInt = $uty;

            fn bits() -> u32 {
                $bits
            }

            fn extract_sign(self) -> (bool, $uty) {
                if self < 0 {
                    (true, !(self as $uty) + 1)
                } else {
                    (false, self as $uty)
                }
            }
        }
    }
}

int_impl!(i32, u32, 32);
int_impl!(i64, u64, 64);
int_impl!(i128, u128, 128);

/// Trait to convert an integer to/from smaller parts
pub trait LargeInt {
    type LowHalf;
    type HighHalf;

    fn low(self) -> Self::LowHalf;
    fn high(self) -> Self::HighHalf;
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
            fn high(self) -> $tyhigh {
                (self >> $halfbits) as $tyhigh
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
