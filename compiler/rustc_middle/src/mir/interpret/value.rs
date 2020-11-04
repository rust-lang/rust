use std::convert::TryFrom;
use std::fmt;

use rustc_apfloat::{
    ieee::{Double, Single},
    Float,
};
use rustc_macros::HashStable;
use rustc_target::abi::{HasDataLayout, Size, TargetDataLayout};

use crate::ty::{ParamEnv, ScalarInt, Ty, TyCtxt};

use super::{AllocId, Allocation, InterpResult, Pointer, PointerArithmetic};

/// Represents the result of const evaluation via the `eval_to_allocation` query.
#[derive(Clone, HashStable, TyEncodable, TyDecodable)]
pub struct ConstAlloc<'tcx> {
    // the value lives here, at offset 0, and that allocation definitely is a `AllocKind::Memory`
    // (so you can use `AllocMap::unwrap_memory`).
    pub alloc_id: AllocId,
    pub ty: Ty<'tcx>,
}

/// Represents a constant value in Rust. `Scalar` and `Slice` are optimizations for
/// array length computations, enum discriminants and the pattern matching logic.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, TyEncodable, TyDecodable, Hash)]
#[derive(HashStable)]
pub enum ConstValue<'tcx> {
    /// Used only for types with `layout::abi::Scalar` ABI and ZSTs.
    ///
    /// Not using the enum `Value` to encode that this must not be `Uninit`.
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

    pub fn try_to_bool(&self) -> Option<bool> {
        match self.try_to_bits(Size::from_bytes(1))? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    pub fn try_to_machine_usize(&self, tcx: TyCtxt<'tcx>) -> Option<u64> {
        Some(self.try_to_bits(tcx.data_layout.pointer_size)? as u64)
    }

    pub fn try_to_bits_for_ty(
        &self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<u128> {
        let size = tcx.layout_of(param_env.with_reveal_all_normalized(tcx).and(ty)).ok()?.size;
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
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, TyEncodable, TyDecodable, Hash)]
#[derive(HashStable)]
pub enum Scalar<Tag = ()> {
    /// The raw bytes of a simple value.
    Int(ScalarInt),

    /// A pointer into an `Allocation`. An `Allocation` in the `memory` module has a list of
    /// relocations, but a `Scalar` is only large enough to contain one, so we just represent the
    /// relocation and its associated offset together as a `Pointer` here.
    Ptr(Pointer<Tag>),
}

#[cfg(target_arch = "x86_64")]
static_assert_size!(Scalar, 24);

// We want the `Debug` output to be readable as it is used by `derive(Debug)` for
// all the Miri types.
impl<Tag: fmt::Debug> fmt::Debug for Scalar<Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Ptr(ptr) => write!(f, "{:?}", ptr),
            Scalar::Int(int) => write!(f, "{:?}", int),
        }
    }
}

impl<Tag: fmt::Debug> fmt::Display for Scalar<Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Ptr(ptr) => write!(f, "pointer to {}", ptr),
            Scalar::Int { .. } => fmt::Debug::fmt(self, f),
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
    /// Tag this scalar with `new_tag` if it is a pointer, leave it unchanged otherwise.
    ///
    /// Used by `MemPlace::replace_tag`.
    #[inline]
    pub fn with_tag<Tag>(self, new_tag: Tag) -> Scalar<Tag> {
        match self {
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.with_tag(new_tag)),
            Scalar::Int(int) => Scalar::Int(int),
        }
    }
}

impl<'tcx, Tag> Scalar<Tag> {
    pub const ZST: Self = Scalar::Int(ScalarInt::ZST);

    /// Erase the tag from the scalar, if any.
    ///
    /// Used by error reporting code to avoid having the error type depend on `Tag`.
    #[inline]
    pub fn erase_tag(self) -> Scalar {
        match self {
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.erase_tag()),
            Scalar::Int(int) => Scalar::Int(int),
        }
    }

    #[inline]
    pub fn null_ptr(cx: &impl HasDataLayout) -> Self {
        Scalar::Int(ScalarInt::null(cx.data_layout().pointer_size))
    }

    #[inline(always)]
    fn ptr_op(
        self,
        dl: &TargetDataLayout,
        f_int: impl FnOnce(u64) -> InterpResult<'tcx, u64>,
        f_ptr: impl FnOnce(Pointer<Tag>) -> InterpResult<'tcx, Pointer<Tag>>,
    ) -> InterpResult<'tcx, Self> {
        match self {
            Scalar::Int(int) => Ok(Scalar::Int(int.ptr_sized_op(dl, f_int)?)),
            Scalar::Ptr(ptr) => Ok(Scalar::Ptr(f_ptr(ptr)?)),
        }
    }

    #[inline]
    pub fn ptr_offset(self, i: Size, cx: &impl HasDataLayout) -> InterpResult<'tcx, Self> {
        let dl = cx.data_layout();
        self.ptr_op(dl, |int| dl.offset(int, i.bytes()), |ptr| ptr.offset(i, dl))
    }

    #[inline]
    pub fn ptr_wrapping_offset(self, i: Size, cx: &impl HasDataLayout) -> Self {
        let dl = cx.data_layout();
        self.ptr_op(
            dl,
            |int| Ok(dl.overflowing_offset(int, i.bytes()).0),
            |ptr| Ok(ptr.wrapping_offset(i, dl)),
        )
        .unwrap()
    }

    #[inline]
    pub fn ptr_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> InterpResult<'tcx, Self> {
        let dl = cx.data_layout();
        self.ptr_op(dl, |int| dl.signed_offset(int, i), |ptr| ptr.signed_offset(i, dl))
    }

    #[inline]
    pub fn ptr_wrapping_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> Self {
        let dl = cx.data_layout();
        self.ptr_op(
            dl,
            |int| Ok(dl.overflowing_signed_offset(int, i).0),
            |ptr| Ok(ptr.wrapping_signed_offset(i, dl)),
        )
        .unwrap()
    }

    #[inline]
    pub fn from_bool(b: bool) -> Self {
        Scalar::Int(b.into())
    }

    #[inline]
    pub fn from_char(c: char) -> Self {
        Scalar::Int(c.into())
    }

    #[inline]
    pub fn try_from_uint(i: impl Into<u128>, size: Size) -> Option<Self> {
        ScalarInt::try_from_uint(i, size).map(Scalar::Int)
    }

    #[inline]
    pub fn from_uint(i: impl Into<u128>, size: Size) -> Self {
        let i = i.into();
        Self::try_from_uint(i, size)
            .unwrap_or_else(|| bug!("Unsigned value {:#x} does not fit in {} bits", i, size.bits()))
    }

    #[inline]
    pub fn from_u8(i: u8) -> Self {
        Scalar::Int(i.into())
    }

    #[inline]
    pub fn from_u16(i: u16) -> Self {
        Scalar::Int(i.into())
    }

    #[inline]
    pub fn from_u32(i: u32) -> Self {
        Scalar::Int(i.into())
    }

    #[inline]
    pub fn from_u64(i: u64) -> Self {
        Scalar::Int(i.into())
    }

    #[inline]
    pub fn from_machine_usize(i: u64, cx: &impl HasDataLayout) -> Self {
        Self::from_uint(i, cx.data_layout().pointer_size)
    }

    #[inline]
    pub fn try_from_int(i: impl Into<i128>, size: Size) -> Option<Self> {
        ScalarInt::try_from_int(i, size).map(Scalar::Int)
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
        Scalar::Int(f.into())
    }

    #[inline]
    pub fn from_f64(f: Double) -> Self {
        Scalar::Int(f.into())
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
            Scalar::Int(int) => Ok(int.assert_bits(target_size)),
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
            Scalar::Int(int) => int.to_bits(target_size).map_err(|size| {
                err_ub!(ScalarSizeMismatch {
                    target_size: target_size.bytes(),
                    data_size: size.bytes(),
                })
                .into()
            }),
            Scalar::Ptr(_) => throw_unsup!(ReadPointerAsBytes),
        }
    }

    #[inline(always)]
    pub fn assert_bits(self, target_size: Size) -> u128 {
        self.to_bits(target_size).expect("expected Raw bits but got a Pointer")
    }

    #[inline]
    pub fn assert_int(self) -> ScalarInt {
        match self {
            Scalar::Ptr(_) => bug!("expected an int but got an abstract pointer"),
            Scalar::Int(int) => int,
        }
    }

    #[inline]
    pub fn assert_ptr(self) -> Pointer<Tag> {
        match self {
            Scalar::Ptr(p) => p,
            Scalar::Int { .. } => bug!("expected a Pointer but got Raw bits"),
        }
    }

    /// Do not call this method!  Dispatch based on the type instead.
    #[inline]
    pub fn is_bits(self) -> bool {
        matches!(self, Scalar::Int { .. })
    }

    /// Do not call this method!  Dispatch based on the type instead.
    #[inline]
    pub fn is_ptr(self) -> bool {
        matches!(self, Scalar::Ptr(_))
    }

    pub fn to_bool(self) -> InterpResult<'tcx, bool> {
        let val = self.to_u8()?;
        match val {
            0 => Ok(false),
            1 => Ok(true),
            _ => throw_ub!(InvalidBool(val)),
        }
    }

    pub fn to_char(self) -> InterpResult<'tcx, char> {
        let val = self.to_u32()?;
        match std::char::from_u32(val) {
            Some(c) => Ok(c),
            None => throw_ub!(InvalidChar(val)),
        }
    }

    #[inline]
    fn to_unsigned_with_bit_width(self, bits: u64) -> InterpResult<'static, u128> {
        let sz = Size::from_bits(bits);
        self.to_bits(sz)
    }

    /// Converts the scalar to produce an `u8`. Fails if the scalar is a pointer.
    pub fn to_u8(self) -> InterpResult<'static, u8> {
        self.to_unsigned_with_bit_width(8).map(|v| u8::try_from(v).unwrap())
    }

    /// Converts the scalar to produce an `u16`. Fails if the scalar is a pointer.
    pub fn to_u16(self) -> InterpResult<'static, u16> {
        self.to_unsigned_with_bit_width(16).map(|v| u16::try_from(v).unwrap())
    }

    /// Converts the scalar to produce an `u32`. Fails if the scalar is a pointer.
    pub fn to_u32(self) -> InterpResult<'static, u32> {
        self.to_unsigned_with_bit_width(32).map(|v| u32::try_from(v).unwrap())
    }

    /// Converts the scalar to produce an `u64`. Fails if the scalar is a pointer.
    pub fn to_u64(self) -> InterpResult<'static, u64> {
        self.to_unsigned_with_bit_width(64).map(|v| u64::try_from(v).unwrap())
    }

    /// Converts the scalar to produce an `u128`. Fails if the scalar is a pointer.
    pub fn to_u128(self) -> InterpResult<'static, u128> {
        self.to_unsigned_with_bit_width(128)
    }

    pub fn to_machine_usize(self, cx: &impl HasDataLayout) -> InterpResult<'static, u64> {
        let b = self.to_bits(cx.data_layout().pointer_size)?;
        Ok(u64::try_from(b).unwrap())
    }

    #[inline]
    fn to_signed_with_bit_width(self, bits: u64) -> InterpResult<'static, i128> {
        let sz = Size::from_bits(bits);
        let b = self.to_bits(sz)?;
        Ok(sz.sign_extend(b) as i128)
    }

    /// Converts the scalar to produce an `i8`. Fails if the scalar is a pointer.
    pub fn to_i8(self) -> InterpResult<'static, i8> {
        self.to_signed_with_bit_width(8).map(|v| i8::try_from(v).unwrap())
    }

    /// Converts the scalar to produce an `i16`. Fails if the scalar is a pointer.
    pub fn to_i16(self) -> InterpResult<'static, i16> {
        self.to_signed_with_bit_width(16).map(|v| i16::try_from(v).unwrap())
    }

    /// Converts the scalar to produce an `i32`. Fails if the scalar is a pointer.
    pub fn to_i32(self) -> InterpResult<'static, i32> {
        self.to_signed_with_bit_width(32).map(|v| i32::try_from(v).unwrap())
    }

    /// Converts the scalar to produce an `i64`. Fails if the scalar is a pointer.
    pub fn to_i64(self) -> InterpResult<'static, i64> {
        self.to_signed_with_bit_width(64).map(|v| i64::try_from(v).unwrap())
    }

    /// Converts the scalar to produce an `i128`. Fails if the scalar is a pointer.
    pub fn to_i128(self) -> InterpResult<'static, i128> {
        self.to_signed_with_bit_width(128)
    }

    pub fn to_machine_isize(self, cx: &impl HasDataLayout) -> InterpResult<'static, i64> {
        let sz = cx.data_layout().pointer_size;
        let b = self.to_bits(sz)?;
        let b = sz.sign_extend(b) as i128;
        Ok(i64::try_from(b).unwrap())
    }

    #[inline]
    pub fn to_f32(self) -> InterpResult<'static, Single> {
        // Going through `u32` to check size and truncation.
        Ok(Single::from_bits(self.to_u32()?.into()))
    }

    #[inline]
    pub fn to_f64(self) -> InterpResult<'static, Double> {
        // Going through `u64` to check size and truncation.
        Ok(Double::from_bits(self.to_u64()?.into()))
    }
}

impl<Tag> From<Pointer<Tag>> for Scalar<Tag> {
    #[inline(always)]
    fn from(ptr: Pointer<Tag>) -> Self {
        Scalar::Ptr(ptr)
    }
}

#[derive(Clone, Copy, Eq, PartialEq, TyEncodable, TyDecodable, HashStable, Hash)]
pub enum ScalarMaybeUninit<Tag = ()> {
    Scalar(Scalar<Tag>),
    Uninit,
}

#[cfg(target_arch = "x86_64")]
static_assert_size!(ScalarMaybeUninit, 24);

impl<Tag> From<Scalar<Tag>> for ScalarMaybeUninit<Tag> {
    #[inline(always)]
    fn from(s: Scalar<Tag>) -> Self {
        ScalarMaybeUninit::Scalar(s)
    }
}

impl<Tag> From<Pointer<Tag>> for ScalarMaybeUninit<Tag> {
    #[inline(always)]
    fn from(s: Pointer<Tag>) -> Self {
        ScalarMaybeUninit::Scalar(s.into())
    }
}

// We want the `Debug` output to be readable as it is used by `derive(Debug)` for
// all the Miri types.
impl<Tag: fmt::Debug> fmt::Debug for ScalarMaybeUninit<Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarMaybeUninit::Uninit => write!(f, "<uninitialized>"),
            ScalarMaybeUninit::Scalar(s) => write!(f, "{:?}", s),
        }
    }
}

impl<Tag: fmt::Debug> fmt::Display for ScalarMaybeUninit<Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarMaybeUninit::Uninit => write!(f, "uninitialized bytes"),
            ScalarMaybeUninit::Scalar(s) => write!(f, "{}", s),
        }
    }
}

impl<'tcx, Tag> ScalarMaybeUninit<Tag> {
    /// Erase the tag from the scalar, if any.
    ///
    /// Used by error reporting code to avoid having the error type depend on `Tag`.
    #[inline]
    pub fn erase_tag(self) -> ScalarMaybeUninit {
        match self {
            ScalarMaybeUninit::Scalar(s) => ScalarMaybeUninit::Scalar(s.erase_tag()),
            ScalarMaybeUninit::Uninit => ScalarMaybeUninit::Uninit,
        }
    }

    #[inline]
    pub fn check_init(self) -> InterpResult<'static, Scalar<Tag>> {
        match self {
            ScalarMaybeUninit::Scalar(scalar) => Ok(scalar),
            ScalarMaybeUninit::Uninit => throw_ub!(InvalidUninitBytes(None)),
        }
    }

    #[inline(always)]
    pub fn to_bool(self) -> InterpResult<'tcx, bool> {
        self.check_init()?.to_bool()
    }

    #[inline(always)]
    pub fn to_char(self) -> InterpResult<'tcx, char> {
        self.check_init()?.to_char()
    }

    #[inline(always)]
    pub fn to_f32(self) -> InterpResult<'tcx, Single> {
        self.check_init()?.to_f32()
    }

    #[inline(always)]
    pub fn to_f64(self) -> InterpResult<'tcx, Double> {
        self.check_init()?.to_f64()
    }

    #[inline(always)]
    pub fn to_u8(self) -> InterpResult<'tcx, u8> {
        self.check_init()?.to_u8()
    }

    #[inline(always)]
    pub fn to_u16(self) -> InterpResult<'tcx, u16> {
        self.check_init()?.to_u16()
    }

    #[inline(always)]
    pub fn to_u32(self) -> InterpResult<'tcx, u32> {
        self.check_init()?.to_u32()
    }

    #[inline(always)]
    pub fn to_u64(self) -> InterpResult<'tcx, u64> {
        self.check_init()?.to_u64()
    }

    #[inline(always)]
    pub fn to_machine_usize(self, cx: &impl HasDataLayout) -> InterpResult<'tcx, u64> {
        self.check_init()?.to_machine_usize(cx)
    }

    #[inline(always)]
    pub fn to_i8(self) -> InterpResult<'tcx, i8> {
        self.check_init()?.to_i8()
    }

    #[inline(always)]
    pub fn to_i16(self) -> InterpResult<'tcx, i16> {
        self.check_init()?.to_i16()
    }

    #[inline(always)]
    pub fn to_i32(self) -> InterpResult<'tcx, i32> {
        self.check_init()?.to_i32()
    }

    #[inline(always)]
    pub fn to_i64(self) -> InterpResult<'tcx, i64> {
        self.check_init()?.to_i64()
    }

    #[inline(always)]
    pub fn to_machine_isize(self, cx: &impl HasDataLayout) -> InterpResult<'tcx, i64> {
        self.check_init()?.to_machine_isize(cx)
    }
}

/// Gets the bytes of a constant slice value.
pub fn get_slice_bytes<'tcx>(cx: &impl HasDataLayout, val: ConstValue<'tcx>) -> &'tcx [u8] {
    if let ConstValue::Slice { data, start, end } = val {
        let len = end - start;
        data.get_bytes(
            cx,
            // invent a pointer, only the offset is relevant anyway
            Pointer::new(AllocId(0), Size::from_bytes(start)),
            Size::from_bytes(len),
        )
        .unwrap_or_else(|err| bug!("const slice is invalid: {:?}", err))
    } else {
        bug!("expected const slice, but found another const value");
    }
}
