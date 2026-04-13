/// Definitions of traits for numeric types
// Implementation based on `num_conv` by jhpratt, under (MIT OR Apache-2.0).

/// Trait for types that this type can be truncated to
#[unstable(feature = "num_internals", reason = "internal implementation detail", issue = "none")]
#[rustc_const_unstable(feature = "integer_extend_truncate", issue = "154330")]
pub const trait TruncateTarget<Target>: crate::sealed::Sealed {
    #[doc(hidden)]
    fn internal_truncate(self) -> Target;

    #[doc(hidden)]
    fn internal_saturating_truncate(self) -> Target;

    #[doc(hidden)]
    fn internal_checked_truncate(self) -> Option<Target>;
}

/// Trait for types that this type can be truncated to
#[unstable(feature = "num_internals", reason = "internal implementation detail", issue = "none")]
#[rustc_const_unstable(feature = "integer_extend_truncate", issue = "154330")]
pub const trait ExtendTarget<Target>: crate::sealed::Sealed {
    #[doc(hidden)]
    fn internal_extend(self) -> Target;
}

macro_rules! impl_truncate {
    ($($from:ty => $($to:ty),+;)*) => {$($(
        const _: () = assert!(
            size_of::<$from>() >= size_of::<$to>(),
            concat!(
                "cannot truncate ",
                stringify!($from),
                " to ",
                stringify!($to),
                " because ",
                stringify!($from),
                " is smaller than ",
                stringify!($to)
            )
        );

        #[unstable(feature = "num_internals", reason = "internal implementation detail", issue = "none")]
        #[rustc_const_unstable(feature = "integer_extend_truncate", issue = "154330")]
        impl const TruncateTarget<$to> for $from {
            #[inline]
            fn internal_truncate(self) -> $to {
                self as _
            }

            #[inline]
            fn internal_saturating_truncate(self) -> $to {
                if self > <$to>::MAX as Self {
                    <$to>::MAX
                } else if self < <$to>::MIN as Self {
                    <$to>::MIN
                } else {
                    self as _
                }
            }

            #[inline]
            fn internal_checked_truncate(self) -> Option<$to> {
                if self > <$to>::MAX as Self || self < <$to>::MIN as Self {
                    None
                } else {
                    Some(self as _)
                }
            }
        }
    )+)*};
}

macro_rules! impl_extend {
    ($($from:ty => $($to:ty),+;)*) => {$($(
        const _: () = assert!(
            size_of::<$from>() <= size_of::<$to>(),
            concat!(
                "cannot extend ",
                stringify!($from),
                " to ",
                stringify!($to),
                " because ",
                stringify!($from),
                " is larger than ",
                stringify!($to)
            )
        );

        #[unstable(feature = "num_internals", reason = "internal implementation detail", issue = "none")]
        #[rustc_const_unstable(feature = "integer_extend_truncate", issue = "154330")]
        impl const ExtendTarget<$to> for $from {
            fn internal_extend(self) -> $to {
                self as _
            }
        }
    )+)*};
}

impl_truncate! {
    u8 => u8;
    u16 => u16, u8;
    u32 => u32, u16, u8;
    u64 => u64, u32, u16, u8;
    u128 => u128, u64, u32, u16, u8;
    usize => usize, u16, u8;

    i8 => i8;
    i16 => i16, i8;
    i32 => i32, i16, i8;
    i64 => i64, i32, i16, i8;
    i128 => i128, i64, i32, i16, i8;
    isize => isize, i16, i8;
}

impl_extend! {
    u8 => u8, u16, u32, u64, u128, usize;
    u16 => u16, u32, u64, u128, usize;
    u32 => u32, u64, u128;
    u64 => u64, u128;
    u128 => u128;
    usize => usize;

    i8 => i8, i16, i32, i64, i128, isize;
    i16 => i16, i32, i64, i128, isize;
    i32 => i32, i64, i128;
    i64 => i64, i128;
    i128 => i128;
    isize => isize;
}
