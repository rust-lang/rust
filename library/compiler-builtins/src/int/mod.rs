
pub mod mul;
pub mod sdiv;
pub mod shift;
pub mod udiv;

/// Trait for some basic operations on integers
pub trait Int {
    /// Returns the bitwidth of the int type
    fn bits() -> u32;
}

macro_rules! int_impl {
    ($ity:ty, $sty:ty, $bits:expr) => {
        impl Int for $ity {
            fn bits() -> u32 {
                $bits
            }
        }
        impl Int for $sty {
            fn bits() -> u32 {
                $bits
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
