use rustc_apfloat::{
    ieee::{Double, Single},
    Float,
};
use rustc_macros::HashStable;
use std::fmt;

use crate::ty::{
    layout::{HasDataLayout, Size},
    ParamEnv, Ty, TyCtxt,
};

use super::{sign_extend, truncate, AllocId, Allocation, InterpResult, Pointer, PointerArithmetic};

/// Represents the result of a raw const operation, pre-validation.
#[derive(Clone, HashStable)]
pub struct RawConst<'tcx> {
    // the value lives here, at offset 0, and that allocation definitely is a `AllocKind::Memory`
    // (so you can use `AllocMap::unwrap_memory`).
    pub alloc_id: AllocId,
    pub ty: Ty<'tcx>,
}

/// Represents a constant value in Rust. `Scalar` and `Slice` are optimizations for
/// array length computations, enum discriminants and the pattern matching logic.
#[derive(
    Copy,
    Clone,
    Debug,
    Eq,
    PartialEq,
    PartialOrd,
    Ord,
    RustcEncodable,
    RustcDecodable,
    Hash,
    HashStable
)]
pub enum ConstValue<'tcx> {
    /// Used only for types with `layout::abi::Scalar` ABI and ZSTs.
    ///
    /// Not using the enum `Value` to encode that this must not be `Undef`.
    Scalar(Scalar),

    /// Used only for `&[u8]` and `&str`
    Slice { data: &'tcx Allocation, start: usize, end: usize },

    /// A value not represented/representable by `Scalar` or `Slice`
    ByRef {
        /// The backing memory of the value, may contain more memory than needed for just the value
        /// in order to share `Allocation`s between values
        alloc: &'tcx Allocation,
        /// Offset into `alloc`
        offset: Size,
    },
}

#[cfg(target_arch = "x86_64")]
static_assert_size!(ConstValue<'_>, 32);

impl<'tcx> ConstValue<'tcx> {
    #[inline]
    pub fn try_to_scalar(&self) -> Option<Scalar> {
        match *self {
            ConstValue::ByRef { .. } | ConstValue::Slice { .. } => None,
            ConstValue::Scalar(val) => Some(val),
        }
    }

    pub fn try_to_bits(&self, size: Size) -> Option<u128> {
        self.try_to_scalar()?.to_bits(size).ok()
    }

    pub fn try_to_bits_for_ty(
        &self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<u128> {
        let size = tcx.layout_of(param_env.with_reveal_all().and(ty)).ok()?.size;
        self.try_to_bits(size)
    }

    pub fn from_bool(b: bool) -> Self {
        ConstValue::Scalar(Scalar::from_bool(b))
    }

    pub fn from_u64(i: u64) -> Self {
        ConstValue::Scalar(Scalar::from_u64(i))
    }

    pub fn from_machine_usize(i: u64, cx: &impl HasDataLayout) -> Self {
        ConstValue::Scalar(Scalar::from_machine_usize(i, cx))
    }
}

/// A `Scalar` represents an immediate, primitive value existing outside of a
/// `memory::Allocation`. It is in many ways like a small chunk of a `Allocation`, up to 8 bytes in
/// size. Like a range of bytes in an `Allocation`, a `Scalar` can either represent the raw bytes
/// of a simple value or a pointer into another `Allocation`
#[derive(
    Clone,
    Copy,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    RustcEncodable,
    RustcDecodable,
    Hash,
    HashStable
)]
pub enum Scalar<Tag = (), Id = AllocId> {
    /// The raw bytes of a simple value.
    Raw {
        /// The first `size` bytes of `data` are the value.
        /// Do not try to read less or more bytes than that. The remaining bytes must be 0.
        data: u128,
        size: u8,
    },

    /// A pointer into an `Allocation`. An `Allocation` in the `memory` module has a list of
    /// relocations, but a `Scalar` is only large enough to contain one, so we just represent the
    /// relocation and its associated offset together as a `Pointer` here.
    Ptr(Pointer<Tag, Id>),
}

#[cfg(target_arch = "x86_64")]
static_assert_size!(Scalar, 24);

impl<Tag: fmt::Debug, Id: fmt::Debug> fmt::Debug for Scalar<Tag, Id> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Ptr(ptr) => write!(f, "{:?}", ptr),
            &Scalar::Raw { data, size } => {
                Scalar::check_data(data, size);
                if size == 0 {
                    write!(f, "<ZST>")
                } else {
                    // Format as hex number wide enough to fit any value of the given `size`.
                    // So data=20, size=1 will be "0x14", but with size=4 it'll be "0x00000014".
                    write!(f, "0x{:>0width$x}", data, width = (size * 2) as usize)
                }
            }
        }
    }
}

impl<Tag> fmt::Display for Scalar<Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Ptr(_) => write!(f, "a pointer"),
            Scalar::Raw { data, .. } => write!(f, "{}", data),
        }
    }
}

impl<Tag> From<Single> for Scalar<Tag> {
    #[inline(always)]
    fn from(f: Single) -> Self {
        Scalar::from_f32(f)
    }
}

impl<Tag> From<Double> for Scalar<Tag> {
    #[inline(always)]
    fn from(f: Double) -> Self {
        Scalar::from_f64(f)
    }
}

impl Scalar<()> {
    /// Make sure the `data` fits in `size`.
    /// This is guaranteed by all constructors here, but since the enum variants are public,
    /// it could still be violated (even though no code outside this file should
    /// construct `Scalar`s).
    #[inline(always)]
    fn check_data(data: u128, size: u8) {
        debug_assert_eq!(
            truncate(data, Size::from_bytes(size as u64)),
            data,
            "Scalar value {:#x} exceeds size of {} bytes",
            data,
            size
        );
    }

    /// Tag this scalar with `new_tag` if it is a pointer, leave it unchanged otherwise.
    ///
    /// Used by `MemPlace::replace_tag`.
    #[inline]
    pub fn with_tag<Tag>(self, new_tag: Tag) -> Scalar<Tag> {
        match self {
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.with_tag(new_tag)),
            Scalar::Raw { data, size } => Scalar::Raw { data, size },
        }
    }
}

impl<'tcx, Tag> Scalar<Tag> {
    /// Erase the tag from the scalar, if any.
    ///
    /// Used by error reporting code to avoid having the error type depend on `Tag`.
    #[inline]
    pub fn erase_tag(self) -> Scalar {
        match self {
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.erase_tag()),
            Scalar::Raw { data, size } => Scalar::Raw { data, size },
        }
    }

    #[inline]
    pub fn ptr_null(cx: &impl HasDataLayout) -> Self {
        Scalar::Raw { data: 0, size: cx.data_layout().pointer_size.bytes() as u8 }
    }

    #[inline]
    pub fn zst() -> Self {
        Scalar::Raw { data: 0, size: 0 }
    }

    #[inline]
    pub fn ptr_offset(self, i: Size, cx: &impl HasDataLayout) -> InterpResult<'tcx, Self> {
        let dl = cx.data_layout();
        match self {
            Scalar::Raw { data, size } => {
                assert_eq!(size as u64, dl.pointer_size.bytes());
                Ok(Scalar::Raw { data: dl.offset(data as u64, i.bytes())? as u128, size })
            }
            Scalar::Ptr(ptr) => ptr.offset(i, dl).map(Scalar::Ptr),
        }
    }

    #[inline]
    pub fn ptr_wrapping_offset(self, i: Size, cx: &impl HasDataLayout) -> Self {
        let dl = cx.data_layout();
        match self {
            Scalar::Raw { data, size } => {
                assert_eq!(size as u64, dl.pointer_size.bytes());
                Scalar::Raw { data: dl.overflowing_offset(data as u64, i.bytes()).0 as u128, size }
            }
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.wrapping_offset(i, dl)),
        }
    }

    #[inline]
    pub fn ptr_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> InterpResult<'tcx, Self> {
        let dl = cx.data_layout();
        match self {
            Scalar::Raw { data, size } => {
                assert_eq!(size as u64, dl.pointer_size().bytes());
                Ok(Scalar::Raw { data: dl.signed_offset(data as u64, i)? as u128, size })
            }
            Scalar::Ptr(ptr) => ptr.signed_offset(i, dl).map(Scalar::Ptr),
        }
    }

    #[inline]
    pub fn ptr_wrapping_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> Self {
        let dl = cx.data_layout();
        match self {
            Scalar::Raw { data, size } => {
                assert_eq!(size as u64, dl.pointer_size.bytes());
                Scalar::Raw {
                    data: dl.overflowing_signed_offset(data as u64, i128::from(i)).0 as u128,
                    size,
                }
            }
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.wrapping_signed_offset(i, dl)),
        }
    }

    #[inline]
    pub fn from_bool(b: bool) -> Self {
        // Guaranteed to be truncated and does not need sign extension.
        Scalar::Raw { data: b as u128, size: 1 }
    }

    #[inline]
    pub fn from_char(c: char) -> Self {
        // Guaranteed to be truncated and does not need sign extension.
        Scalar::Raw { data: c as u128, size: 4 }
    }

    #[inline]
    pub fn try_from_uint(i: impl Into<u128>, size: Size) -> Option<Self> {
        let i = i.into();
        if truncate(i, size) == i {
            Some(Scalar::Raw { data: i, size: size.bytes() as u8 })
        } else {
            None
        }
    }

    #[inline]
    pub fn from_uint(i: impl Into<u128>, size: Size) -> Self {
        let i = i.into();
        Self::try_from_uint(i, size)
            .unwrap_or_else(|| bug!("Unsigned value {:#x} does not fit in {} bits", i, size.bits()))
    }

    #[inline]
    pub fn from_u8(i: u8) -> Self {
        // Guaranteed to be truncated and does not need sign extension.
        Scalar::Raw { data: i as u128, size: 1 }
    }

    #[inline]
    pub fn from_u16(i: u16) -> Self {
        // Guaranteed to be truncated and does not need sign extension.
        Scalar::Raw { data: i as u128, size: 2 }
    }

    #[inline]
    pub fn from_u32(i: u32) -> Self {
        // Guaranteed to be truncated and does not need sign extension.
        Scalar::Raw { data: i as u128, size: 4 }
    }

    #[inline]
    pub fn from_u64(i: u64) -> Self {
        // Guaranteed to be truncated and does not need sign extension.
        Scalar::Raw { data: i as u128, size: 8 }
    }

    #[inline]
    pub fn from_machine_usize(i: u64, cx: &impl HasDataLayout) -> Self {
        Self::from_uint(i, cx.data_layout().pointer_size)
    }

    #[inline]
    pub fn try_from_int(i: impl Into<i128>, size: Size) -> Option<Self> {
        let i = i.into();
        // `into` performed sign extension, we have to truncate
        let truncated = truncate(i as u128, size);
        if sign_extend(truncated, size) as i128 == i {
            Some(Scalar::Raw { data: truncated, size: size.bytes() as u8 })
        } else {
            None
        }
    }

    #[inline]
    pub fn from_int(i: impl Into<i128>, size: Size) -> Self {
        let i = i.into();
        Self::try_from_int(i, size)
            .unwrap_or_else(|| bug!("Signed value {:#x} does not fit in {} bits", i, size.bits()))
    }

    #[inline]
    pub fn from_i8(i: i8) -> Self {
        Self::from_int(i, Size::from_bits(8))
    }

    #[inline]
    pub fn from_i16(i: i16) -> Self {
        Self::from_int(i, Size::from_bits(16))
    }

    #[inline]
    pub fn from_i32(i: i32) -> Self {
        Self::from_int(i, Size::from_bits(32))
    }

    #[inline]
    pub fn from_i64(i: i64) -> Self {
        Self::from_int(i, Size::from_bits(64))
    }

    #[inline]
    pub fn from_machine_isize(i: i64, cx: &impl HasDataLayout) -> Self {
        Self::from_int(i, cx.data_layout().pointer_size)
    }

    #[inline]
    pub fn from_f32(f: Single) -> Self {
        // We trust apfloat to give us properly truncated data.
        Scalar::Raw { data: f.to_bits(), size: 4 }
    }

    #[inline]
    pub fn from_f64(f: Double) -> Self {
        // We trust apfloat to give us properly truncated data.
        Scalar::Raw { data: f.to_bits(), size: 8 }
    }

    /// This is very rarely the method you want!  You should dispatch on the type
    /// and use `force_bits`/`assert_bits`/`force_ptr`/`assert_ptr`.
    /// This method only exists for the benefit of low-level memory operations
    /// as well as the implementation of the `force_*` methods.
    #[inline]
    pub fn to_bits_or_ptr(
        self,
        target_size: Size,
        cx: &impl HasDataLayout,
    ) -> Result<u128, Pointer<Tag>> {
        assert_ne!(target_size.bytes(), 0, "you should never look at the bits of a ZST");
        match self {
            Scalar::Raw { data, size } => {
                assert_eq!(target_size.bytes(), size as u64);
                Scalar::check_data(data, size);
                Ok(data)
            }
            Scalar::Ptr(ptr) => {
                assert_eq!(target_size, cx.data_layout().pointer_size);
                Err(ptr)
            }
        }
    }

    /// This method is intentionally private!
    /// It is just a helper for other methods in this file.
    #[inline]
    fn to_bits(self, target_size: Size) -> InterpResult<'tcx, u128> {
        assert_ne!(target_size.bytes(), 0, "you should never look at the bits of a ZST");
        match self {
            Scalar::Raw { data, size } => {
                assert_eq!(target_size.bytes(), size as u64);
                Scalar::check_data(data, size);
                Ok(data)
            }
            Scalar::Ptr(_) => throw_unsup!(ReadPointerAsBytes),
        }
    }

    #[inline(always)]
    pub fn assert_bits(self, target_size: Size) -> u128 {
        self.to_bits(target_size).expect("expected Raw bits but got a Pointer")
    }

    #[inline]
    pub fn assert_ptr(self) -> Pointer<Tag> {
        match self {
            Scalar::Ptr(p) => p,
            Scalar::Raw { .. } => bug!("expected a Pointer but got Raw bits"),
        }
    }

    /// Do not call this method!  Dispatch based on the type instead.
    #[inline]
    pub fn is_bits(self) -> bool {
        match self {
            Scalar::Raw { .. } => true,
            _ => false,
        }
    }

    /// Do not call this method!  Dispatch based on the type instead.
    #[inline]
    pub fn is_ptr(self) -> bool {
        match self {
            Scalar::Ptr(_) => true,
            _ => false,
        }
    }

    pub fn to_bool(self) -> InterpResult<'tcx, bool> {
        match self {
            Scalar::Raw { data: 0, size: 1 } => Ok(false),
            Scalar::Raw { data: 1, size: 1 } => Ok(true),
            _ => throw_unsup!(InvalidBool),
        }
    }

    pub fn to_char(self) -> InterpResult<'tcx, char> {
        let val = self.to_u32()?;
        match ::std::char::from_u32(val) {
            Some(c) => Ok(c),
            None => throw_unsup!(InvalidChar(val as u128)),
        }
    }

    #[inline]
    fn to_unsigned_with_bit_width(self, bits: u64) -> InterpResult<'static, u128> {
        let sz = Size::from_bits(bits);
        self.to_bits(sz)
    }

    /// Converts the scalar to produce an `u8`. Fails if the scalar is a pointer.
    pub fn to_u8(self) -> InterpResult<'static, u8> {
        self.to_unsigned_with_bit_width(8).map(|v| v as u8)
    }

    /// Converts the scalar to produce an `u16`. Fails if the scalar is a pointer.
    pub fn to_u16(self) -> InterpResult<'static, u16> {
        self.to_unsigned_with_bit_width(16).map(|v| v as u16)
    }

    /// Converts the scalar to produce an `u32`. Fails if the scalar is a pointer.
    pub fn to_u32(self) -> InterpResult<'static, u32> {
        self.to_unsigned_with_bit_width(32).map(|v| v as u32)
    }

    /// Converts the scalar to produce an `u64`. Fails if the scalar is a pointer.
    pub fn to_u64(self) -> InterpResult<'static, u64> {
        self.to_unsigned_with_bit_width(64).map(|v| v as u64)
    }

    pub fn to_machine_usize(self, cx: &impl HasDataLayout) -> InterpResult<'static, u64> {
        let b = self.to_bits(cx.data_layout().pointer_size)?;
        Ok(b as u64)
    }

    #[inline]
    fn to_signed_with_bit_width(self, bits: u64) -> InterpResult<'static, i128> {
        let sz = Size::from_bits(bits);
        let b = self.to_bits(sz)?;
        Ok(sign_extend(b, sz) as i128)
    }

    /// Converts the scalar to produce an `i8`. Fails if the scalar is a pointer.
    pub fn to_i8(self) -> InterpResult<'static, i8> {
        self.to_signed_with_bit_width(8).map(|v| v as i8)
    }

    /// Converts the scalar to produce an `i16`. Fails if the scalar is a pointer.
    pub fn to_i16(self) -> InterpResult<'static, i16> {
        self.to_signed_with_bit_width(16).map(|v| v as i16)
    }

    /// Converts the scalar to produce an `i32`. Fails if the scalar is a pointer.
    pub fn to_i32(self) -> InterpResult<'static, i32> {
        self.to_signed_with_bit_width(32).map(|v| v as i32)
    }

    /// Converts the scalar to produce an `i64`. Fails if the scalar is a pointer.
    pub fn to_i64(self) -> InterpResult<'static, i64> {
        self.to_signed_with_bit_width(64).map(|v| v as i64)
    }

    pub fn to_machine_isize(self, cx: &impl HasDataLayout) -> InterpResult<'static, i64> {
        let sz = cx.data_layout().pointer_size;
        let b = self.to_bits(sz)?;
        let b = sign_extend(b, sz) as i128;
        Ok(b as i64)
    }

    #[inline]
    pub fn to_f32(self) -> InterpResult<'static, Single> {
        // Going through `u32` to check size and truncation.
        Ok(Single::from_bits(self.to_u32()? as u128))
    }

    #[inline]
    pub fn to_f64(self) -> InterpResult<'static, Double> {
        // Going through `u64` to check size and truncation.
        Ok(Double::from_bits(self.to_u64()? as u128))
    }
}

impl<Tag> From<Pointer<Tag>> for Scalar<Tag> {
    #[inline(always)]
    fn from(ptr: Pointer<Tag>) -> Self {
        Scalar::Ptr(ptr)
    }
}

#[derive(Clone, Copy, Eq, PartialEq, RustcEncodable, RustcDecodable, HashStable, Hash)]
pub enum ScalarMaybeUndef<Tag = (), Id = AllocId> {
    Scalar(Scalar<Tag, Id>),
    Undef,
}

impl<Tag> From<Scalar<Tag>> for ScalarMaybeUndef<Tag> {
    #[inline(always)]
    fn from(s: Scalar<Tag>) -> Self {
        ScalarMaybeUndef::Scalar(s)
    }
}

impl<Tag> From<Pointer<Tag>> for ScalarMaybeUndef<Tag> {
    #[inline(always)]
    fn from(s: Pointer<Tag>) -> Self {
        ScalarMaybeUndef::Scalar(s.into())
    }
}

impl<Tag: fmt::Debug, Id: fmt::Debug> fmt::Debug for ScalarMaybeUndef<Tag, Id> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarMaybeUndef::Undef => write!(f, "Undef"),
            ScalarMaybeUndef::Scalar(s) => write!(f, "{:?}", s),
        }
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

impl<'tcx, Tag> ScalarMaybeUndef<Tag> {
    /// Erase the tag from the scalar, if any.
    ///
    /// Used by error reporting code to avoid having the error type depend on `Tag`.
    #[inline]
    pub fn erase_tag(self) -> ScalarMaybeUndef {
        match self {
            ScalarMaybeUndef::Scalar(s) => ScalarMaybeUndef::Scalar(s.erase_tag()),
            ScalarMaybeUndef::Undef => ScalarMaybeUndef::Undef,
        }
    }

    #[inline]
    pub fn not_undef(self) -> InterpResult<'static, Scalar<Tag>> {
        match self {
            ScalarMaybeUndef::Scalar(scalar) => Ok(scalar),
            ScalarMaybeUndef::Undef => throw_unsup!(ReadUndefBytes(Size::ZERO)),
        }
    }

    #[inline(always)]
    pub fn to_bool(self) -> InterpResult<'tcx, bool> {
        self.not_undef()?.to_bool()
    }

    #[inline(always)]
    pub fn to_char(self) -> InterpResult<'tcx, char> {
        self.not_undef()?.to_char()
    }

    #[inline(always)]
    pub fn to_f32(self) -> InterpResult<'tcx, Single> {
        self.not_undef()?.to_f32()
    }

    #[inline(always)]
    pub fn to_f64(self) -> InterpResult<'tcx, Double> {
        self.not_undef()?.to_f64()
    }

    #[inline(always)]
    pub fn to_u8(self) -> InterpResult<'tcx, u8> {
        self.not_undef()?.to_u8()
    }

    #[inline(always)]
    pub fn to_u16(self) -> InterpResult<'tcx, u16> {
        self.not_undef()?.to_u16()
    }

    #[inline(always)]
    pub fn to_u32(self) -> InterpResult<'tcx, u32> {
        self.not_undef()?.to_u32()
    }

    #[inline(always)]
    pub fn to_u64(self) -> InterpResult<'tcx, u64> {
        self.not_undef()?.to_u64()
    }

    #[inline(always)]
    pub fn to_machine_usize(self, cx: &impl HasDataLayout) -> InterpResult<'tcx, u64> {
        self.not_undef()?.to_machine_usize(cx)
    }

    #[inline(always)]
    pub fn to_i8(self) -> InterpResult<'tcx, i8> {
        self.not_undef()?.to_i8()
    }

    #[inline(always)]
    pub fn to_i16(self) -> InterpResult<'tcx, i16> {
        self.not_undef()?.to_i16()
    }

    #[inline(always)]
    pub fn to_i32(self) -> InterpResult<'tcx, i32> {
        self.not_undef()?.to_i32()
    }

    #[inline(always)]
    pub fn to_i64(self) -> InterpResult<'tcx, i64> {
        self.not_undef()?.to_i64()
    }

    #[inline(always)]
    pub fn to_machine_isize(self, cx: &impl HasDataLayout) -> InterpResult<'tcx, i64> {
        self.not_undef()?.to_machine_isize(cx)
    }
}

/// Gets the bytes of a constant slice value.
pub fn get_slice_bytes<'tcx>(cx: &impl HasDataLayout, val: ConstValue<'tcx>) -> &'tcx [u8] {
    if let ConstValue::Slice { data, start, end } = val {
        let len = end - start;
        data.get_bytes(
            cx,
            // invent a pointer, only the offset is relevant anyway
            Pointer::new(AllocId(0), Size::from_bytes(start as u64)),
            Size::from_bytes(len as u64),
        )
        .unwrap_or_else(|err| bug!("const slice is invalid: {:?}", err))
    } else {
        bug!("expected const slice, but found another const value");
    }
}
