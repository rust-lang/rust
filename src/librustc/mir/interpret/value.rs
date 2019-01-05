use std::fmt;

use crate::ty::{Ty, layout::{HasDataLayout, Size}};

use super::{EvalResult, Pointer, PointerArithmetic, Allocation, AllocId, sign_extend, truncate};

/// Represents the result of a raw const operation, pre-validation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, RustcEncodable, RustcDecodable, Hash)]
pub struct RawConst<'tcx> {
    // the value lives here, at offset 0, and that allocation definitely is a `AllocKind::Memory`
    // (so you can use `AllocMap::unwrap_memory`).
    pub alloc_id: AllocId,
    pub ty: Ty<'tcx>,
}

/// Represents a constant value in Rust. Scalar and ScalarPair are optimizations which
/// matches the LocalValue optimizations for easy conversions between Value and ConstValue.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash)]
pub enum ConstValue<'tcx> {
    /// Used only for types with layout::abi::Scalar ABI and ZSTs
    ///
    /// Not using the enum `Value` to encode that this must not be `Undef`
    Scalar(Scalar),

    /// Used only for *fat pointers* with layout::abi::ScalarPair
    ///
    /// Needed for pattern matching code related to slices and strings.
    ScalarPair(Scalar, Scalar),

    /// An allocation + offset into the allocation.
    /// Invariant: The AllocId matches the allocation.
    ByRef(AllocId, &'tcx Allocation, Size),
}

impl<'tcx> ConstValue<'tcx> {
    #[inline]
    pub fn try_to_scalar(&self) -> Option<Scalar> {
        match *self {
            ConstValue::ByRef(..) |
            ConstValue::ScalarPair(..) => None,
            ConstValue::Scalar(val) => Some(val),
        }
    }

    #[inline]
    pub fn try_to_bits(&self, size: Size) -> Option<u128> {
        self.try_to_scalar()?.to_bits(size).ok()
    }

    #[inline]
    pub fn try_to_ptr(&self) -> Option<Pointer> {
        self.try_to_scalar()?.to_ptr().ok()
    }

    #[inline]
    pub fn new_slice(
        val: Scalar,
        len: u64,
        cx: &impl HasDataLayout
    ) -> Self {
        ConstValue::ScalarPair(val, Scalar::Bits {
            bits: len as u128,
            size: cx.data_layout().pointer_size.bytes() as u8,
        })
    }

    #[inline]
    pub fn new_dyn_trait(val: Scalar, vtable: Pointer) -> Self {
        ConstValue::ScalarPair(val, Scalar::Ptr(vtable))
    }
}

/// A `Scalar` represents an immediate, primitive value existing outside of a
/// `memory::Allocation`. It is in many ways like a small chunk of a `Allocation`, up to 8 bytes in
/// size. Like a range of bytes in an `Allocation`, a `Scalar` can either represent the raw bytes
/// of a simple value or a pointer into another `Allocation`
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, RustcEncodable, RustcDecodable, Hash)]
pub enum Scalar<Tag=(), Id=AllocId> {
    /// The raw bytes of a simple value.
    Bits {
        /// The first `size` bytes are the value.
        /// Do not try to read less or more bytes than that. The remaining bytes must be 0.
        size: u8,
        bits: u128,
    },

    /// A pointer into an `Allocation`. An `Allocation` in the `memory` module has a list of
    /// relocations, but a `Scalar` is only large enough to contain one, so we just represent the
    /// relocation and its associated offset together as a `Pointer` here.
    Ptr(Pointer<Tag, Id>),
}

impl<Tag> fmt::Display for Scalar<Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Ptr(_) => write!(f, "a pointer"),
            Scalar::Bits { bits, .. } => write!(f, "{}", bits),
        }
    }
}

impl<'tcx> Scalar<()> {
    #[inline]
    pub fn with_default_tag<Tag>(self) -> Scalar<Tag>
        where Tag: Default
    {
        match self {
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.with_default_tag()),
            Scalar::Bits { bits, size } => Scalar::Bits { bits, size },
        }
    }
}

impl<'tcx, Tag> Scalar<Tag> {
    #[inline]
    pub fn erase_tag(self) -> Scalar {
        match self {
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.erase_tag()),
            Scalar::Bits { bits, size } => Scalar::Bits { bits, size },
        }
    }

    #[inline]
    pub fn with_tag(self, new_tag: Tag) -> Self {
        match self {
            Scalar::Ptr(ptr) => Scalar::Ptr(Pointer { tag: new_tag, ..ptr }),
            Scalar::Bits { bits, size } => Scalar::Bits { bits, size },
        }
    }

    #[inline]
    pub fn ptr_null(cx: &impl HasDataLayout) -> Self {
        Scalar::Bits {
            bits: 0,
            size: cx.data_layout().pointer_size.bytes() as u8,
        }
    }

    #[inline]
    pub fn zst() -> Self {
        Scalar::Bits { bits: 0, size: 0 }
    }

    #[inline]
    pub fn ptr_offset(self, i: Size, cx: &impl HasDataLayout) -> EvalResult<'tcx, Self> {
        let dl = cx.data_layout();
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, dl.pointer_size.bytes());
                Ok(Scalar::Bits {
                    bits: dl.offset(bits as u64, i.bytes())? as u128,
                    size,
                })
            }
            Scalar::Ptr(ptr) => ptr.offset(i, dl).map(Scalar::Ptr),
        }
    }

    #[inline]
    pub fn ptr_wrapping_offset(self, i: Size, cx: &impl HasDataLayout) -> Self {
        let dl = cx.data_layout();
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, dl.pointer_size.bytes());
                Scalar::Bits {
                    bits: dl.overflowing_offset(bits as u64, i.bytes()).0 as u128,
                    size,
                }
            }
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.wrapping_offset(i, dl)),
        }
    }

    #[inline]
    pub fn ptr_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> EvalResult<'tcx, Self> {
        let dl = cx.data_layout();
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, dl.pointer_size().bytes());
                Ok(Scalar::Bits {
                    bits: dl.signed_offset(bits as u64, i)? as u128,
                    size,
                })
            }
            Scalar::Ptr(ptr) => ptr.signed_offset(i, dl).map(Scalar::Ptr),
        }
    }

    #[inline]
    pub fn ptr_wrapping_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> Self {
        let dl = cx.data_layout();
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, dl.pointer_size.bytes());
                Scalar::Bits {
                    bits: dl.overflowing_signed_offset(bits as u64, i128::from(i)).0 as u128,
                    size,
                }
            }
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.wrapping_signed_offset(i, dl)),
        }
    }

    /// Returns this pointers offset from the allocation base, or from NULL (for
    /// integer pointers).
    #[inline]
    pub fn get_ptr_offset(self, cx: &impl HasDataLayout) -> Size {
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, cx.pointer_size().bytes());
                Size::from_bytes(bits as u64)
            }
            Scalar::Ptr(ptr) => ptr.offset,
        }
    }

    #[inline]
    pub fn is_null_ptr(self, cx: &impl HasDataLayout) -> bool {
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, cx.data_layout().pointer_size.bytes());
                bits == 0
            },
            Scalar::Ptr(_) => false,
        }
    }

    #[inline]
    pub fn from_bool(b: bool) -> Self {
        Scalar::Bits { bits: b as u128, size: 1 }
    }

    #[inline]
    pub fn from_char(c: char) -> Self {
        Scalar::Bits { bits: c as u128, size: 4 }
    }

    #[inline]
    pub fn from_uint(i: impl Into<u128>, size: Size) -> Self {
        let i = i.into();
        debug_assert_eq!(truncate(i, size), i,
                         "Unsigned value {} does not fit in {} bits", i, size.bits());
        Scalar::Bits { bits: i, size: size.bytes() as u8 }
    }

    #[inline]
    pub fn from_int(i: impl Into<i128>, size: Size) -> Self {
        let i = i.into();
        // `into` performed sign extension, we have to truncate
        let truncated = truncate(i as u128, size);
        debug_assert_eq!(sign_extend(truncated, size) as i128, i,
                         "Signed value {} does not fit in {} bits", i, size.bits());
        Scalar::Bits { bits: truncated, size: size.bytes() as u8 }
    }

    #[inline]
    pub fn from_f32(f: f32) -> Self {
        Scalar::Bits { bits: f.to_bits() as u128, size: 4 }
    }

    #[inline]
    pub fn from_f64(f: f64) -> Self {
        Scalar::Bits { bits: f.to_bits() as u128, size: 8 }
    }

    #[inline]
    pub fn to_bits(self, target_size: Size) -> EvalResult<'tcx, u128> {
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(target_size.bytes(), size as u64);
                assert_ne!(size, 0, "to_bits cannot be used with zsts");
                Ok(bits)
            }
            Scalar::Ptr(_) => err!(ReadPointerAsBytes),
        }
    }

    #[inline]
    pub fn to_ptr(self) -> EvalResult<'tcx, Pointer<Tag>> {
        match self {
            Scalar::Bits { bits: 0, .. } => err!(InvalidNullPointerUsage),
            Scalar::Bits { .. } => err!(ReadBytesAsPointer),
            Scalar::Ptr(p) => Ok(p),
        }
    }

    #[inline]
    pub fn is_bits(self) -> bool {
        match self {
            Scalar::Bits { .. } => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_ptr(self) -> bool {
        match self {
            Scalar::Ptr(_) => true,
            _ => false,
        }
    }

    pub fn to_bool(self) -> EvalResult<'tcx, bool> {
        match self {
            Scalar::Bits { bits: 0, size: 1 } => Ok(false),
            Scalar::Bits { bits: 1, size: 1 } => Ok(true),
            _ => err!(InvalidBool),
        }
    }

    pub fn to_char(self) -> EvalResult<'tcx, char> {
        let val = self.to_u32()?;
        match ::std::char::from_u32(val) {
            Some(c) => Ok(c),
            None => err!(InvalidChar(val as u128)),
        }
    }

    pub fn to_u8(self) -> EvalResult<'static, u8> {
        let sz = Size::from_bits(8);
        let b = self.to_bits(sz)?;
        assert_eq!(b as u8 as u128, b);
        Ok(b as u8)
    }

    pub fn to_u32(self) -> EvalResult<'static, u32> {
        let sz = Size::from_bits(32);
        let b = self.to_bits(sz)?;
        assert_eq!(b as u32 as u128, b);
        Ok(b as u32)
    }

    pub fn to_u64(self) -> EvalResult<'static, u64> {
        let sz = Size::from_bits(64);
        let b = self.to_bits(sz)?;
        assert_eq!(b as u64 as u128, b);
        Ok(b as u64)
    }

    pub fn to_usize(self, cx: &impl HasDataLayout) -> EvalResult<'static, u64> {
        let b = self.to_bits(cx.data_layout().pointer_size)?;
        assert_eq!(b as u64 as u128, b);
        Ok(b as u64)
    }

    pub fn to_i8(self) -> EvalResult<'static, i8> {
        let sz = Size::from_bits(8);
        let b = self.to_bits(sz)?;
        let b = sign_extend(b, sz) as i128;
        assert_eq!(b as i8 as i128, b);
        Ok(b as i8)
    }

    pub fn to_i32(self) -> EvalResult<'static, i32> {
        let sz = Size::from_bits(32);
        let b = self.to_bits(sz)?;
        let b = sign_extend(b, sz) as i128;
        assert_eq!(b as i32 as i128, b);
        Ok(b as i32)
    }

    pub fn to_i64(self) -> EvalResult<'static, i64> {
        let sz = Size::from_bits(64);
        let b = self.to_bits(sz)?;
        let b = sign_extend(b, sz) as i128;
        assert_eq!(b as i64 as i128, b);
        Ok(b as i64)
    }

    pub fn to_isize(self, cx: &impl HasDataLayout) -> EvalResult<'static, i64> {
        let b = self.to_bits(cx.data_layout().pointer_size)?;
        let b = sign_extend(b, cx.data_layout().pointer_size) as i128;
        assert_eq!(b as i64 as i128, b);
        Ok(b as i64)
    }

    #[inline]
    pub fn to_f32(self) -> EvalResult<'static, f32> {
        Ok(f32::from_bits(self.to_u32()?))
    }

    #[inline]
    pub fn to_f64(self) -> EvalResult<'static, f64> {
        Ok(f64::from_bits(self.to_u64()?))
    }
}

impl<Tag> From<Pointer<Tag>> for Scalar<Tag> {
    #[inline(always)]
    fn from(ptr: Pointer<Tag>) -> Self {
        Scalar::Ptr(ptr)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, RustcEncodable, RustcDecodable, Hash)]
pub enum ScalarMaybeUndef<Tag=(), Id=AllocId> {
    Scalar(Scalar<Tag, Id>),
    Undef,
}

impl<Tag> From<Scalar<Tag>> for ScalarMaybeUndef<Tag> {
    #[inline(always)]
    fn from(s: Scalar<Tag>) -> Self {
        ScalarMaybeUndef::Scalar(s)
    }
}

impl<Tag> fmt::Display for ScalarMaybeUndef<Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarMaybeUndef::Undef => write!(f, "uninitialized bytes"),
            ScalarMaybeUndef::Scalar(s) => write!(f, "{}", s),
        }
    }
}

impl<'tcx> ScalarMaybeUndef<()> {
    #[inline]
    pub fn with_default_tag<Tag>(self) -> ScalarMaybeUndef<Tag>
        where Tag: Default
    {
        match self {
            ScalarMaybeUndef::Scalar(s) => ScalarMaybeUndef::Scalar(s.with_default_tag()),
            ScalarMaybeUndef::Undef => ScalarMaybeUndef::Undef,
        }
    }
}

impl<'tcx, Tag> ScalarMaybeUndef<Tag> {
    #[inline]
    pub fn erase_tag(self) -> ScalarMaybeUndef
    {
        match self {
            ScalarMaybeUndef::Scalar(s) => ScalarMaybeUndef::Scalar(s.erase_tag()),
            ScalarMaybeUndef::Undef => ScalarMaybeUndef::Undef,
        }
    }

    #[inline]
    pub fn not_undef(self) -> EvalResult<'static, Scalar<Tag>> {
        match self {
            ScalarMaybeUndef::Scalar(scalar) => Ok(scalar),
            ScalarMaybeUndef::Undef => err!(ReadUndefBytes(Size::from_bytes(0))),
        }
    }

    #[inline(always)]
    pub fn to_ptr(self) -> EvalResult<'tcx, Pointer<Tag>> {
        self.not_undef()?.to_ptr()
    }

    #[inline(always)]
    pub fn to_bits(self, target_size: Size) -> EvalResult<'tcx, u128> {
        self.not_undef()?.to_bits(target_size)
    }

    #[inline(always)]
    pub fn to_bool(self) -> EvalResult<'tcx, bool> {
        self.not_undef()?.to_bool()
    }

    #[inline(always)]
    pub fn to_char(self) -> EvalResult<'tcx, char> {
        self.not_undef()?.to_char()
    }

    #[inline(always)]
    pub fn to_f32(self) -> EvalResult<'tcx, f32> {
        self.not_undef()?.to_f32()
    }

    #[inline(always)]
    pub fn to_f64(self) -> EvalResult<'tcx, f64> {
        self.not_undef()?.to_f64()
    }

    #[inline(always)]
    pub fn to_u8(self) -> EvalResult<'tcx, u8> {
        self.not_undef()?.to_u8()
    }

    #[inline(always)]
    pub fn to_u32(self) -> EvalResult<'tcx, u32> {
        self.not_undef()?.to_u32()
    }

    #[inline(always)]
    pub fn to_u64(self) -> EvalResult<'tcx, u64> {
        self.not_undef()?.to_u64()
    }

    #[inline(always)]
    pub fn to_usize(self, cx: &impl HasDataLayout) -> EvalResult<'tcx, u64> {
        self.not_undef()?.to_usize(cx)
    }

    #[inline(always)]
    pub fn to_i8(self) -> EvalResult<'tcx, i8> {
        self.not_undef()?.to_i8()
    }

    #[inline(always)]
    pub fn to_i32(self) -> EvalResult<'tcx, i32> {
        self.not_undef()?.to_i32()
    }

    #[inline(always)]
    pub fn to_i64(self) -> EvalResult<'tcx, i64> {
        self.not_undef()?.to_i64()
    }

    #[inline(always)]
    pub fn to_isize(self, cx: &impl HasDataLayout) -> EvalResult<'tcx, i64> {
        self.not_undef()?.to_isize(cx)
    }
}

impl_stable_hash_for!(enum ::mir::interpret::ScalarMaybeUndef {
    Scalar(v),
    Undef
});
