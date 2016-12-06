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

// TODO: Once i128/u128 support lands, we'll want to add impls for those as well
impl Int for u32 {
    type OtherSign = i32;
    type UnsignedInt = u32;

    fn bits() -> u32 {
        32
    }

    fn extract_sign(self) -> (bool, u32) {
        (false, self)
    }
}

impl Int for i32 {
    type OtherSign = u32;
    type UnsignedInt = u32;

    fn bits() -> u32 {
        32
    }

    fn extract_sign(self) -> (bool, u32) {
        if self < 0 {
            (true, !(self as u32) + 1)
        } else {
            (false, self as u32)
        }
    }
}

impl Int for u64 {
    type OtherSign = i64;
    type UnsignedInt = u64;

    fn bits() -> u32 {
        64
    }

    fn extract_sign(self) -> (bool, u64) {
        (false, self)
    }
}

impl Int for i64 {
    type OtherSign = u64;
    type UnsignedInt = u64;

    fn bits() -> u32 {
        64
    }

    fn extract_sign(self) -> (bool, u64) {
        if self < 0 {
            (true, !(self as u64) + 1)
        } else {
            (false, self as u64)
        }
    }
}

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
