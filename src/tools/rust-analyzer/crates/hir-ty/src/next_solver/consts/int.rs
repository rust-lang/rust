use std::{fmt, hash::Hash, num::NonZero};

use rustc_abi::{Size, TargetDataLayout};

/// The raw bytes of a simple value.
///
/// This is a packed struct in order to allow this type to be optimally embedded in enums
/// (like Scalar).
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
#[repr(Rust, packed)]
pub struct ScalarInt {
    /// The first `size` bytes of `data` are the value.
    /// Do not try to read less or more bytes than that. The remaining bytes must be 0.
    data: u128,
    size: NonZero<u8>,
}

impl ScalarInt {
    pub const TRUE: ScalarInt = ScalarInt { data: 1_u128, size: NonZero::new(1).unwrap() };
    pub const FALSE: ScalarInt = ScalarInt { data: 0_u128, size: NonZero::new(1).unwrap() };

    fn raw(data: u128, size: Size) -> Self {
        Self { data, size: NonZero::new(size.bytes() as u8).unwrap() }
    }

    #[inline]
    pub fn size(self) -> Size {
        Size::from_bytes(self.size.get())
    }

    /// Make sure the `data` fits in `size`.
    /// This is guaranteed by all constructors here, but having had this check saved us from
    /// bugs many times in the past, so keeping it around is definitely worth it.
    #[inline(always)]
    fn check_data(self) {
        // Using a block `{self.data}` here to force a copy instead of using `self.data`
        // directly, because `debug_assert_eq` takes references to its arguments and formatting
        // arguments and would thus borrow `self.data`. Since `Self`
        // is a packed struct, that would create a possibly unaligned reference, which
        // is UB.
        debug_assert_eq!(
            self.size().truncate(self.data),
            { self.data },
            "Scalar value {:#x} exceeds size of {} bytes",
            { self.data },
            self.size
        );
    }

    #[inline]
    pub fn null(size: Size) -> Self {
        Self::raw(0, size)
    }

    #[inline]
    pub fn is_null(self) -> bool {
        self.data == 0
    }

    #[inline]
    pub fn try_from_uint(i: impl Into<u128>, size: Size) -> Option<Self> {
        let (r, overflow) = Self::truncate_from_uint(i, size);
        if overflow { None } else { Some(r) }
    }

    /// Returns the truncated result, and whether truncation changed the value.
    #[inline]
    pub fn truncate_from_uint(i: impl Into<u128>, size: Size) -> (Self, bool) {
        let data = i.into();
        let r = Self::raw(size.truncate(data), size);
        (r, r.data != data)
    }

    #[inline]
    pub fn try_from_int(i: impl Into<i128>, size: Size) -> Option<Self> {
        let (r, overflow) = Self::truncate_from_int(i, size);
        if overflow { None } else { Some(r) }
    }

    /// Returns the truncated result, and whether truncation changed the value.
    #[inline]
    pub fn truncate_from_int(i: impl Into<i128>, size: Size) -> (Self, bool) {
        let data = i.into();
        // `into` performed sign extension, we have to truncate
        let r = Self::raw(size.truncate(data as u128), size);
        (r, size.sign_extend(r.data) != data)
    }

    #[inline]
    pub fn try_from_target_usize(
        i: impl Into<u128>,
        data_layout: &TargetDataLayout,
    ) -> Option<Self> {
        Self::try_from_uint(i, data_layout.pointer_size())
    }

    /// Try to convert this ScalarInt to the raw underlying bits.
    /// Fails if the size is wrong. Generally a wrong size should lead to a panic,
    /// but Miri sometimes wants to be resilient to size mismatches,
    /// so the interpreter will generally use this `try` method.
    #[inline]
    pub fn try_to_bits(self, target_size: Size) -> Result<u128, Size> {
        assert_ne!(target_size.bytes(), 0, "you should never look at the bits of a ZST");
        if target_size.bytes() == u64::from(self.size.get()) {
            self.check_data();
            Ok(self.data)
        } else {
            Err(self.size())
        }
    }

    #[inline]
    pub fn to_bits(self, target_size: Size) -> u128 {
        self.try_to_bits(target_size).unwrap_or_else(|size| {
            panic!("expected int of size {}, but got size {}", target_size.bytes(), size.bytes())
        })
    }

    /// Extracts the bits from the scalar without checking the size.
    #[inline]
    pub fn to_bits_unchecked(self) -> u128 {
        self.check_data();
        self.data
    }

    /// Converts the `ScalarInt` to an unsigned integer of the given size.
    /// Panics if the size of the `ScalarInt` is not equal to `size`.
    #[inline]
    pub fn to_uint(self, size: Size) -> u128 {
        self.to_bits(size)
    }

    #[inline]
    pub fn to_uint_unchecked(self) -> u128 {
        self.data
    }

    /// Converts the `ScalarInt` to `u8`.
    /// Panics if the `size` of the `ScalarInt`in not equal to 1 byte.
    #[inline]
    pub fn to_u8(self) -> u8 {
        self.to_uint(Size::from_bits(8)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to `u16`.
    /// Panics if the size of the `ScalarInt` in not equal to 2 bytes.
    #[inline]
    pub fn to_u16(self) -> u16 {
        self.to_uint(Size::from_bits(16)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to `u32`.
    /// Panics if the `size` of the `ScalarInt` in not equal to 4 bytes.
    #[inline]
    pub fn to_u32(self) -> u32 {
        self.to_uint(Size::from_bits(32)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to `u64`.
    /// Panics if the `size` of the `ScalarInt` in not equal to 8 bytes.
    #[inline]
    pub fn to_u64(self) -> u64 {
        self.to_uint(Size::from_bits(64)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to `u128`.
    /// Panics if the `size` of the `ScalarInt` in not equal to 16 bytes.
    #[inline]
    pub fn to_u128(self) -> u128 {
        self.to_uint(Size::from_bits(128))
    }

    #[inline]
    pub fn to_target_usize(&self, data_layout: &TargetDataLayout) -> u64 {
        self.to_uint(data_layout.pointer_size()).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to `bool`.
    /// Panics if the `size` of the `ScalarInt` is not equal to 1 byte.
    /// Errors if it is not a valid `bool`.
    #[inline]
    pub fn try_to_bool(self) -> Result<bool, ()> {
        match self.to_u8() {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(()),
        }
    }

    /// Converts the `ScalarInt` to a signed integer of the given size.
    /// Panics if the size of the `ScalarInt` is not equal to `size`.
    #[inline]
    pub fn to_int(self, size: Size) -> i128 {
        let b = self.to_bits(size);
        size.sign_extend(b)
    }

    #[inline]
    pub fn to_int_unchecked(self) -> i128 {
        self.size().sign_extend(self.data)
    }

    /// Converts the `ScalarInt` to i8.
    /// Panics if the size of the `ScalarInt` is not equal to 1 byte.
    pub fn to_i8(self) -> i8 {
        self.to_int(Size::from_bits(8)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to i16.
    /// Panics if the size of the `ScalarInt` is not equal to 2 bytes.
    pub fn to_i16(self) -> i16 {
        self.to_int(Size::from_bits(16)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to i32.
    /// Panics if the size of the `ScalarInt` is not equal to 4 bytes.
    pub fn to_i32(self) -> i32 {
        self.to_int(Size::from_bits(32)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to i64.
    /// Panics if the size of the `ScalarInt` is not equal to 8 bytes.
    pub fn to_i64(self) -> i64 {
        self.to_int(Size::from_bits(64)).try_into().unwrap()
    }

    /// Converts the `ScalarInt` to i128.
    /// Panics if the size of the `ScalarInt` is not equal to 16 bytes.
    pub fn to_i128(self) -> i128 {
        self.to_int(Size::from_bits(128))
    }

    #[inline]
    pub fn to_target_isize(&self, data_layout: &TargetDataLayout) -> i64 {
        self.to_int(data_layout.pointer_size()).try_into().unwrap()
    }
}

macro_rules! from_x_for_scalar_int {
    ($($ty:ty),*) => {
        $(
            impl From<$ty> for ScalarInt {
                #[inline]
                fn from(u: $ty) -> Self {
                    Self {
                        data: u128::from(u),
                        size: NonZero::new(size_of::<$ty>() as u8).unwrap(),
                    }
                }
            }
        )*
    }
}

macro_rules! from_scalar_int_for_x {
    ($($ty:ty),*) => {
        $(
            impl From<ScalarInt> for $ty {
                #[inline]
                fn from(int: ScalarInt) -> Self {
                    // The `unwrap` cannot fail because to_uint (if it succeeds)
                    // is guaranteed to return a value that fits into the size.
                    int.to_uint(Size::from_bytes(size_of::<$ty>()))
                       .try_into().unwrap()
                }
            }
        )*
    }
}

from_x_for_scalar_int!(u8, u16, u32, u64, u128, bool);
from_scalar_int_for_x!(u8, u16, u32, u64, u128);

impl TryFrom<ScalarInt> for bool {
    type Error = ();
    #[inline]
    fn try_from(int: ScalarInt) -> Result<Self, ()> {
        int.try_to_bool()
    }
}

impl From<char> for ScalarInt {
    #[inline]
    fn from(c: char) -> Self {
        (c as u32).into()
    }
}

macro_rules! from_x_for_scalar_int_signed {
    ($($ty:ty),*) => {
        $(
            impl From<$ty> for ScalarInt {
                #[inline]
                fn from(u: $ty) -> Self {
                    Self {
                        data: u128::from(u.cast_unsigned()), // go via the unsigned type of the same size
                        size: NonZero::new(size_of::<$ty>() as u8).unwrap(),
                    }
                }
            }
        )*
    }
}

macro_rules! from_scalar_int_for_x_signed {
    ($($ty:ty),*) => {
        $(
            impl From<ScalarInt> for $ty {
                #[inline]
                fn from(int: ScalarInt) -> Self {
                    // The `unwrap` cannot fail because to_int (if it succeeds)
                    // is guaranteed to return a value that fits into the size.
                    int.to_int(Size::from_bytes(size_of::<$ty>()))
                       .try_into().unwrap()
                }
            }
        )*
    }
}

from_x_for_scalar_int_signed!(i8, i16, i32, i64, i128);
from_scalar_int_for_x_signed!(i8, i16, i32, i64, i128);

impl From<std::cmp::Ordering> for ScalarInt {
    #[inline]
    fn from(c: std::cmp::Ordering) -> Self {
        // Here we rely on `cmp::Ordering` having the same values in host and target!
        ScalarInt::from(c as i8)
    }
}

/// Error returned when a conversion from ScalarInt to char fails.
#[derive(Debug)]
pub struct CharTryFromScalarInt;

impl TryFrom<ScalarInt> for char {
    type Error = CharTryFromScalarInt;

    #[inline]
    fn try_from(int: ScalarInt) -> Result<Self, Self::Error> {
        match char::from_u32(int.to_u32()) {
            Some(c) => Ok(c),
            None => Err(CharTryFromScalarInt),
        }
    }
}

impl fmt::Debug for ScalarInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Dispatch to LowerHex below.
        write!(f, "0x{self:x}")
    }
}

impl fmt::LowerHex for ScalarInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.check_data();
        if f.alternate() {
            // Like regular ints, alternate flag adds leading `0x`.
            write!(f, "0x")?;
        }
        // Format as hex number wide enough to fit any value of the given `size`.
        // So data=20, size=1 will be "0x14", but with size=4 it'll be "0x00000014".
        // Using a block `{self.data}` here to force a copy instead of using `self.data`
        // directly, because `write!` takes references to its formatting arguments and
        // would thus borrow `self.data`. Since `Self`
        // is a packed struct, that would create a possibly unaligned reference, which
        // is UB.
        write!(f, "{:01$x}", { self.data }, self.size.get() as usize * 2)
    }
}

impl fmt::UpperHex for ScalarInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.check_data();
        // Format as hex number wide enough to fit any value of the given `size`.
        // So data=20, size=1 will be "0x14", but with size=4 it'll be "0x00000014".
        // Using a block `{self.data}` here to force a copy instead of using `self.data`
        // directly, because `write!` takes references to its formatting arguments and
        // would thus borrow `self.data`. Since `Self`
        // is a packed struct, that would create a possibly unaligned reference, which
        // is UB.
        write!(f, "{:01$X}", { self.data }, self.size.get() as usize * 2)
    }
}

impl fmt::Display for ScalarInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.check_data();
        write!(f, "{}", { self.data })
    }
}
