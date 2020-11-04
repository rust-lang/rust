use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::Float;
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
        let raw = int.data;
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
            let max = Size::from_bytes(size).truncate(u128::MAX);
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

/// The raw bytes of a simple value.
///
/// This is a packed struct in order to allow this type to be optimally embedded in enums
/// (like Scalar).
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(packed)]
pub struct ScalarInt {
    /// The first `size` bytes of `data` are the value.
    /// Do not try to read less or more bytes than that. The remaining bytes must be 0.
    data: u128,
    size: u8,
}

// Cannot derive these, as the derives take references to the fields, and we
// can't take references to fields of packed structs.
impl<CTX> crate::ty::HashStable<CTX> for ScalarInt {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut crate::ty::StableHasher) {
        // Using a block `{self.data}` here to force a copy instead of using `self.data`
        // directly, because `hash_stable` takes `&self` and would thus borrow `self.data`.
        // Since `Self` is a packed struct, that would create a possibly unaligned reference,
        // which is UB.
        { self.data }.hash_stable(hcx, hasher);
        self.size.hash_stable(hcx, hasher);
    }
}

impl<S: Encoder> Encodable<S> for ScalarInt {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u128(self.data)?;
        s.emit_u8(self.size)
    }
}

impl<D: Decoder> Decodable<D> for ScalarInt {
    fn decode(d: &mut D) -> Result<ScalarInt, D::Error> {
        Ok(ScalarInt { data: d.read_u128()?, size: d.read_u8()? })
    }
}

impl ScalarInt {
    pub const TRUE: ScalarInt = ScalarInt { data: 1_u128, size: 1 };

    pub const FALSE: ScalarInt = ScalarInt { data: 0_u128, size: 1 };

    pub const ZST: ScalarInt = ScalarInt { data: 0_u128, size: 0 };

    #[inline]
    pub fn size(self) -> Size {
        Size::from_bytes(self.size)
    }

    /// Make sure the `data` fits in `size`.
    /// This is guaranteed by all constructors here, but having had this check saved us from
    /// bugs many times in the past, so keeping it around is definitely worth it.
    #[inline(always)]
    fn check_data(self) {
        // Using a block `{self.data}` here to force a copy instead of using `self.data`
        // directly, because `assert_eq` takes references to its arguments and formatting
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
        Self { data: 0, size: size.bytes() as u8 }
    }

    #[inline]
    pub fn is_null(self) -> bool {
        self.data == 0
    }

    pub(crate) fn ptr_sized_op<E>(
        self,
        dl: &TargetDataLayout,
        f_int: impl FnOnce(u64) -> Result<u64, E>,
    ) -> Result<Self, E> {
        assert_eq!(u64::from(self.size), dl.pointer_size.bytes());
        Ok(Self::try_from_uint(f_int(u64::try_from(self.data).unwrap())?, self.size()).unwrap())
    }

    #[inline]
    pub fn try_from_uint(i: impl Into<u128>, size: Size) -> Option<Self> {
        let data = i.into();
        if size.truncate(data) == data {
            Some(Self { data, size: size.bytes() as u8 })
        } else {
            None
        }
    }

    #[inline]
    pub fn try_from_int(i: impl Into<i128>, size: Size) -> Option<Self> {
        let i = i.into();
        // `into` performed sign extension, we have to truncate
        let truncated = size.truncate(i as u128);
        if size.sign_extend(truncated) as i128 == i {
            Some(Self { data: truncated, size: size.bytes() as u8 })
        } else {
            None
        }
    }

    #[inline]
    pub fn assert_bits(self, target_size: Size) -> u128 {
        self.to_bits(target_size).unwrap_or_else(|size| {
            bug!("expected int of size {}, but got size {}", target_size.bytes(), size.bytes())
        })
    }

    #[inline]
    pub fn to_bits(self, target_size: Size) -> Result<u128, Size> {
        assert_ne!(target_size.bytes(), 0, "you should never look at the bits of a ZST");
        if target_size.bytes() == u64::from(self.size) {
            self.check_data();
            Ok(self.data)
        } else {
            Err(self.size())
        }
    }
}

macro_rules! from {
    ($($ty:ty),*) => {
        $(
            impl From<$ty> for ScalarInt {
                #[inline]
                fn from(u: $ty) -> Self {
                    Self {
                        data: u128::from(u),
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
                type Error = Size;
                #[inline]
                fn try_from(int: ScalarInt) -> Result<Self, Size> {
                    // The `unwrap` cannot fail because to_bits (if it succeeds)
                    // is guaranteed to return a value that fits into the size.
                    int.to_bits(Size::from_bytes(std::mem::size_of::<$ty>()))
                       .map(|u| u.try_into().unwrap())
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
        Self { data: c as u128, size: std::mem::size_of::<char>() as u8 }
    }
}

impl TryFrom<ScalarInt> for char {
    type Error = Size;
    #[inline]
    fn try_from(int: ScalarInt) -> Result<Self, Size> {
        int.to_bits(Size::from_bytes(std::mem::size_of::<char>()))
            .map(|u| char::from_u32(u.try_into().unwrap()).unwrap())
    }
}

impl From<Single> for ScalarInt {
    #[inline]
    fn from(f: Single) -> Self {
        // We trust apfloat to give us properly truncated data.
        Self { data: f.to_bits(), size: 4 }
    }
}

impl TryFrom<ScalarInt> for Single {
    type Error = Size;
    #[inline]
    fn try_from(int: ScalarInt) -> Result<Self, Size> {
        int.to_bits(Size::from_bytes(4)).map(Self::from_bits)
    }
}

impl From<Double> for ScalarInt {
    #[inline]
    fn from(f: Double) -> Self {
        // We trust apfloat to give us properly truncated data.
        Self { data: f.to_bits(), size: 8 }
    }
}

impl TryFrom<ScalarInt> for Double {
    type Error = Size;
    #[inline]
    fn try_from(int: ScalarInt) -> Result<Self, Size> {
        int.to_bits(Size::from_bytes(8)).map(Self::from_bits)
    }
}

impl fmt::Debug for ScalarInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.size == 0 {
            self.check_data();
            write!(f, "<ZST>")
        } else {
            // Dispatch to LowerHex below.
            write!(f, "0x{:x}", self)
        }
    }
}

impl fmt::LowerHex for ScalarInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.check_data();
        // Format as hex number wide enough to fit any value of the given `size`.
        // So data=20, size=1 will be "0x14", but with size=4 it'll be "0x00000014".
        // Using a block `{self.data}` here to force a copy instead of using `self.data`
        // directly, because `write!` takes references to its formatting arguments and
        // would thus borrow `self.data`. Since `Self`
        // is a packed struct, that would create a possibly unaligned reference, which
        // is UB.
        write!(f, "{:01$x}", { self.data }, self.size as usize * 2)
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
        write!(f, "{:01$X}", { self.data }, self.size as usize * 2)
    }
}
