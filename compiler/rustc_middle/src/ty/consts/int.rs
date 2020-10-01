use crate::mir::interpret::{sign_extend, truncate, InterpErrorInfo, InterpResult};
use crate::throw_ub;
use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::Float;
use rustc_macros::HashStable;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use rustc_target::abi::{Size, TargetDataLayout};
use std::convert::{TryFrom, TryInto};
use std::fmt;

#[derive(Copy, Clone)]
/// A type for representing any integer. Only used for printing.
pub struct ConstInt {
    /// The "untyped" variant of `ConstInt`.
    int: ScalarInt,
    /// Whether the value is of a signed integer type.
    signed: bool,
    /// Whether the value is a `usize` or `isize` type.
    is_ptr_sized_integral: bool,
}

impl ConstInt {
    pub fn new(int: ScalarInt, signed: bool, is_ptr_sized_integral: bool) -> Self {
        Self { int, signed, is_ptr_sized_integral }
    }
}

impl std::fmt::Debug for ConstInt {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { int, signed, is_ptr_sized_integral } = *self;
        let size = int.size().bytes();
        let raw = int.data();
        if signed {
            let bit_size = size * 8;
            let min = 1u128 << (bit_size - 1);
            let max = min - 1;
            if raw == min {
                match (size, is_ptr_sized_integral) {
                    (_, true) => write!(fmt, "isize::MIN"),
                    (1, _) => write!(fmt, "i8::MIN"),
                    (2, _) => write!(fmt, "i16::MIN"),
                    (4, _) => write!(fmt, "i32::MIN"),
                    (8, _) => write!(fmt, "i64::MIN"),
                    (16, _) => write!(fmt, "i128::MIN"),
                    _ => bug!("ConstInt 0x{:x} with size = {} and signed = {}", raw, size, signed),
                }
            } else if raw == max {
                match (size, is_ptr_sized_integral) {
                    (_, true) => write!(fmt, "isize::MAX"),
                    (1, _) => write!(fmt, "i8::MAX"),
                    (2, _) => write!(fmt, "i16::MAX"),
                    (4, _) => write!(fmt, "i32::MAX"),
                    (8, _) => write!(fmt, "i64::MAX"),
                    (16, _) => write!(fmt, "i128::MAX"),
                    _ => bug!("ConstInt 0x{:x} with size = {} and signed = {}", raw, size, signed),
                }
            } else {
                match size {
                    1 => write!(fmt, "{}", raw as i8)?,
                    2 => write!(fmt, "{}", raw as i16)?,
                    4 => write!(fmt, "{}", raw as i32)?,
                    8 => write!(fmt, "{}", raw as i64)?,
                    16 => write!(fmt, "{}", raw as i128)?,
                    _ => bug!("ConstInt 0x{:x} with size = {} and signed = {}", raw, size, signed),
                }
                if fmt.alternate() {
                    match (size, is_ptr_sized_integral) {
                        (_, true) => write!(fmt, "_isize")?,
                        (1, _) => write!(fmt, "_i8")?,
                        (2, _) => write!(fmt, "_i16")?,
                        (4, _) => write!(fmt, "_i32")?,
                        (8, _) => write!(fmt, "_i64")?,
                        (16, _) => write!(fmt, "_i128")?,
                        _ => bug!(),
                    }
                }
                Ok(())
            }
        } else {
            let max = truncate(u128::MAX, Size::from_bytes(size));
            if raw == max {
                match (size, is_ptr_sized_integral) {
                    (_, true) => write!(fmt, "usize::MAX"),
                    (1, _) => write!(fmt, "u8::MAX"),
                    (2, _) => write!(fmt, "u16::MAX"),
                    (4, _) => write!(fmt, "u32::MAX"),
                    (8, _) => write!(fmt, "u64::MAX"),
                    (16, _) => write!(fmt, "u128::MAX"),
                    _ => bug!("ConstInt 0x{:x} with size = {} and signed = {}", raw, size, signed),
                }
            } else {
                match size {
                    1 => write!(fmt, "{}", raw as u8)?,
                    2 => write!(fmt, "{}", raw as u16)?,
                    4 => write!(fmt, "{}", raw as u32)?,
                    8 => write!(fmt, "{}", raw as u64)?,
                    16 => write!(fmt, "{}", raw as u128)?,
                    _ => bug!("ConstInt 0x{:x} with size = {} and signed = {}", raw, size, signed),
                }
                if fmt.alternate() {
                    match (size, is_ptr_sized_integral) {
                        (_, true) => write!(fmt, "_usize")?,
                        (1, _) => write!(fmt, "_u8")?,
                        (2, _) => write!(fmt, "_u16")?,
                        (4, _) => write!(fmt, "_u32")?,
                        (8, _) => write!(fmt, "_u64")?,
                        (16, _) => write!(fmt, "_u128")?,
                        _ => bug!(),
                    }
                }
                Ok(())
            }
        }
    }
}

// FIXME: reuse in `super::int::ConstInt` and `Scalar::Bits`
/// The raw bytes of a simple value.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[derive(HashStable)]
pub struct ScalarInt {
    /// The first `size` bytes of `data` are the value.
    /// Do not try to read less or more bytes than that. The remaining bytes must be 0.
    ///
    /// This is an array in order to allow this type to be optimally embedded in enums
    /// (like Scalar).
    bytes: [u8; 16],
    size: u8,
}

impl<S: Encoder> Encodable<S> for ScalarInt {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u128(self.data())?;
        s.emit_u8(self.size)
    }
}

impl<D: Decoder> Decodable<D> for ScalarInt {
    fn decode(d: &mut D) -> Result<ScalarInt, D::Error> {
        Ok(ScalarInt { bytes: d.read_u128()?.to_ne_bytes(), size: d.read_u8()? })
    }
}

impl ScalarInt {
    pub const TRUE: ScalarInt = ScalarInt { bytes: 1_u128.to_ne_bytes(), size: 1 };

    pub const FALSE: ScalarInt = ScalarInt { bytes: 0_u128.to_ne_bytes(), size: 1 };

    pub const ZST: ScalarInt = ScalarInt { bytes: 0_u128.to_ne_bytes(), size: 0 };

    fn data(self) -> u128 {
        u128::from_ne_bytes(self.bytes)
    }

    #[inline]
    pub fn size(self) -> Size {
        Size::from_bytes(self.size)
    }

    /// Make sure the `data` fits in `size`.
    /// This is guaranteed by all constructors here, but since the enum variants are public,
    /// it could still be violated (even though no code outside this file should
    /// construct `Scalar`s).
    #[inline(always)]
    fn check_data(self) {
        debug_assert_eq!(
            truncate(self.data(), self.size()),
            self.data(),
            "Scalar value {:#x} exceeds size of {} bytes",
            self.data(),
            self.size
        );
    }

    #[inline]
    pub fn zst() -> Self {
        Self::null(Size::ZERO)
    }

    #[inline]
    pub fn null(size: Size) -> Self {
        Self { bytes: [0; 16], size: size.bytes() as u8 }
    }

    pub(crate) fn ptr_sized_op<'tcx>(
        self,
        dl: &TargetDataLayout,
        f_int: impl FnOnce(u64) -> InterpResult<'tcx, u64>,
    ) -> InterpResult<'tcx, Self> {
        assert_eq!(u64::from(self.size), dl.pointer_size.bytes());
        Ok(Self {
            bytes: u128::from(f_int(u64::try_from(self.data()).unwrap())?).to_ne_bytes(),
            size: self.size,
        })
    }

    #[inline]
    pub fn try_from_uint(i: impl Into<u128>, size: Size) -> Option<Self> {
        let data = i.into();
        if truncate(data, size) == data {
            Some(Self { bytes: data.to_ne_bytes(), size: size.bytes() as u8 })
        } else {
            None
        }
    }

    #[inline]
    pub fn try_from_int(i: impl Into<i128>, size: Size) -> Option<Self> {
        let i = i.into();
        // `into` performed sign extension, we have to truncate
        let truncated = truncate(i as u128, size);
        if sign_extend(truncated, size) as i128 == i {
            Some(Self { bytes: truncated.to_ne_bytes(), size: size.bytes() as u8 })
        } else {
            None
        }
    }

    #[inline]
    pub fn assert_bits(self, target_size: Size) -> u128 {
        assert_ne!(target_size.bytes(), 0, "you should never look at the bits of a ZST");
        assert_eq!(target_size.bytes(), u64::from(self.size));
        self.check_data();
        self.data()
    }

    #[inline]
    pub fn to_bits(self, target_size: Size) -> InterpResult<'static, u128> {
        assert_ne!(target_size.bytes(), 0, "you should never look at the bits of a ZST");
        if target_size.bytes() != u64::from(self.size) {
            throw_ub!(ScalarSizeMismatch {
                target_size: target_size.bytes(),
                data_size: u64::from(self.size),
            });
        }
        self.check_data();
        Ok(self.data())
    }
}

macro_rules! from {
    ($($ty:ty),*) => {
        $(
            impl From<$ty> for ScalarInt {
                #[inline]
                fn from(u: $ty) -> Self {
                    Self {
                        bytes: u128::from(u).to_ne_bytes(),
                        size: std::mem::size_of::<$ty>() as u8,
                    }
                }
            }
        )*
    }
}

macro_rules! try_from {
    ($($ty:ty),*) => {
        $(
            impl TryFrom<ScalarInt> for $ty {
                type Error = InterpErrorInfo<'static>;
                #[inline]
                fn try_from(int: ScalarInt) -> InterpResult<'static, Self> {
                    int.to_bits(Size::from_bytes(std::mem::size_of::<$ty>())).map(|u| u.try_into().unwrap())
                }
            }
        )*
    }
}

from!(u8, u16, u32, u64, u128, bool);
try_from!(u8, u16, u32, u64, u128);

impl From<char> for ScalarInt {
    #[inline]
    fn from(c: char) -> Self {
        Self { bytes: (c as u128).to_ne_bytes(), size: std::mem::size_of::<char>() as u8 }
    }
}

impl TryFrom<ScalarInt> for char {
    type Error = InterpErrorInfo<'static>;
    #[inline]
    fn try_from(int: ScalarInt) -> InterpResult<'static, Self> {
        int.to_bits(Size::from_bytes(std::mem::size_of::<char>()))
            .map(|u| char::from_u32(u.try_into().unwrap()).unwrap())
    }
}

impl From<Single> for ScalarInt {
    #[inline]
    fn from(f: Single) -> Self {
        // We trust apfloat to give us properly truncated data.
        Self { bytes: f.to_bits().to_ne_bytes(), size: 4 }
    }
}

impl TryFrom<ScalarInt> for Single {
    type Error = InterpErrorInfo<'static>;
    #[inline]
    fn try_from(int: ScalarInt) -> InterpResult<'static, Self> {
        int.to_bits(Size::from_bytes(4)).map(Self::from_bits)
    }
}

impl From<Double> for ScalarInt {
    #[inline]
    fn from(f: Double) -> Self {
        // We trust apfloat to give us properly truncated data.
        Self { bytes: f.to_bits().to_ne_bytes(), size: 8 }
    }
}

impl TryFrom<ScalarInt> for Double {
    type Error = InterpErrorInfo<'static>;
    #[inline]
    fn try_from(int: ScalarInt) -> InterpResult<'static, Self> {
        int.to_bits(Size::from_bytes(8)).map(Self::from_bits)
    }
}

impl fmt::Debug for ScalarInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.size == 0 {
            self.check_data();
            write!(f, "<ZST>")
        } else {
            write!(f, "0x{:x}", self)
        }
    }
}

impl fmt::LowerHex for ScalarInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.check_data();
        // Format as hex number wide enough to fit any value of the given `size`.
        // So data=20, size=1 will be "0x14", but with size=4 it'll be "0x00000014".
        write!(f, "{:01$x}", self.data(), self.size as usize * 2)
    }
}
