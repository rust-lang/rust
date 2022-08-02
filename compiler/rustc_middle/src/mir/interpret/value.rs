use std::convert::{TryFrom, TryInto};
use std::fmt;

use rustc_apfloat::{
    ieee::{Double, Single},
    Float,
};
use rustc_macros::HashStable;
use rustc_target::abi::{HasDataLayout, Size};

use crate::ty::{Lift, ParamEnv, ScalarInt, Ty, TyCtxt};

use super::{
    AllocId, AllocRange, ConstAllocation, InterpResult, Pointer, PointerArithmetic, Provenance,
    ScalarSizeMismatch,
};

/// Represents the result of const evaluation via the `eval_to_allocation` query.
#[derive(Copy, Clone, HashStable, TyEncodable, TyDecodable, Debug, Hash, Eq, PartialEq)]
pub struct ConstAlloc<'tcx> {
    // the value lives here, at offset 0, and that allocation definitely is an `AllocKind::Memory`
    // (so you can use `AllocMap::unwrap_memory`).
    pub alloc_id: AllocId,
    pub ty: Ty<'tcx>,
}

/// Represents a constant value in Rust. `Scalar` and `Slice` are optimizations for
/// array length computations, enum discriminants and the pattern matching logic.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, TyEncodable, TyDecodable, Hash)]
#[derive(HashStable)]
pub enum ConstValue<'tcx> {
    /// Used only for types with `layout::abi::Scalar` ABI.
    ///
    /// Not using the enum `Value` to encode that this must not be `Uninit`.
    Scalar(Scalar),

    /// Only used for ZSTs.
    ZeroSized,

    /// Used only for `&[u8]` and `&str`
    Slice { data: ConstAllocation<'tcx>, start: usize, end: usize },

    /// A value not represented/representable by `Scalar` or `Slice`
    ByRef {
        /// The backing memory of the value, may contain more memory than needed for just the value
        /// in order to share `ConstAllocation`s between values
        alloc: ConstAllocation<'tcx>,
        /// Offset into `alloc`
        offset: Size,
    },
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(ConstValue<'_>, 32);

impl<'a, 'tcx> Lift<'tcx> for ConstValue<'a> {
    type Lifted = ConstValue<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<ConstValue<'tcx>> {
        Some(match self {
            ConstValue::Scalar(s) => ConstValue::Scalar(s),
            ConstValue::ZeroSized => ConstValue::ZeroSized,
            ConstValue::Slice { data, start, end } => {
                ConstValue::Slice { data: tcx.lift(data)?, start, end }
            }
            ConstValue::ByRef { alloc, offset } => {
                ConstValue::ByRef { alloc: tcx.lift(alloc)?, offset }
            }
        })
    }
}

impl<'tcx> ConstValue<'tcx> {
    #[inline]
    pub fn try_to_scalar(&self) -> Option<Scalar<AllocId>> {
        match *self {
            ConstValue::ByRef { .. } | ConstValue::Slice { .. } | ConstValue::ZeroSized => None,
            ConstValue::Scalar(val) => Some(val),
        }
    }

    pub fn try_to_scalar_int(&self) -> Option<ScalarInt> {
        self.try_to_scalar()?.try_to_int().ok()
    }

    pub fn try_to_bits(&self, size: Size) -> Option<u128> {
        self.try_to_scalar_int()?.to_bits(size).ok()
    }

    pub fn try_to_bool(&self) -> Option<bool> {
        self.try_to_scalar_int()?.try_into().ok()
    }

    pub fn try_to_machine_usize(&self, tcx: TyCtxt<'tcx>) -> Option<u64> {
        self.try_to_scalar_int()?.try_to_machine_usize(tcx).ok()
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
/// `memory::Allocation`. It is in many ways like a small chunk of an `Allocation`, up to 16 bytes in
/// size. Like a range of bytes in an `Allocation`, a `Scalar` can either represent the raw bytes
/// of a simple value or a pointer into another `Allocation`
///
/// These variants would be private if there was a convenient way to achieve that in Rust.
/// Do *not* match on a `Scalar`! Use the various `to_*` methods instead.
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, TyEncodable, TyDecodable, Hash)]
#[derive(HashStable)]
pub enum Scalar<Prov = AllocId> {
    /// The raw bytes of a simple value.
    Int(ScalarInt),

    /// A pointer into an `Allocation`. An `Allocation` in the `memory` module has a list of
    /// relocations, but a `Scalar` is only large enough to contain one, so we just represent the
    /// relocation and its associated offset together as a `Pointer` here.
    ///
    /// We also store the size of the pointer, such that a `Scalar` always knows how big it is.
    /// The size is always the pointer size of the current target, but this is not information
    /// that we always have readily available.
    Ptr(Pointer<Prov>, u8),
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(Scalar, 24);

// We want the `Debug` output to be readable as it is used by `derive(Debug)` for
// all the Miri types.
impl<Prov: Provenance> fmt::Debug for Scalar<Prov> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Ptr(ptr, _size) => write!(f, "{:?}", ptr),
            Scalar::Int(int) => write!(f, "{:?}", int),
        }
    }
}

impl<Prov: Provenance> fmt::Display for Scalar<Prov> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Ptr(ptr, _size) => write!(f, "pointer to {:?}", ptr),
            Scalar::Int(int) => write!(f, "{}", int),
        }
    }
}

impl<Prov: Provenance> fmt::LowerHex for Scalar<Prov> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Ptr(ptr, _size) => write!(f, "pointer to {:?}", ptr),
            Scalar::Int(int) => write!(f, "{:#x}", int),
        }
    }
}

impl<Prov> From<Single> for Scalar<Prov> {
    #[inline(always)]
    fn from(f: Single) -> Self {
        Scalar::from_f32(f)
    }
}

impl<Prov> From<Double> for Scalar<Prov> {
    #[inline(always)]
    fn from(f: Double) -> Self {
        Scalar::from_f64(f)
    }
}

impl<Prov> From<ScalarInt> for Scalar<Prov> {
    #[inline(always)]
    fn from(ptr: ScalarInt) -> Self {
        Scalar::Int(ptr)
    }
}

impl<Prov> Scalar<Prov> {
    #[inline(always)]
    pub fn from_pointer(ptr: Pointer<Prov>, cx: &impl HasDataLayout) -> Self {
        Scalar::Ptr(ptr, u8::try_from(cx.pointer_size().bytes()).unwrap())
    }

    /// Create a Scalar from a pointer with an `Option<_>` provenance (where `None` represents a
    /// plain integer / "invalid" pointer).
    pub fn from_maybe_pointer(ptr: Pointer<Option<Prov>>, cx: &impl HasDataLayout) -> Self {
        match ptr.into_parts() {
            (Some(prov), offset) => Scalar::from_pointer(Pointer::new(prov, offset), cx),
            (None, offset) => {
                Scalar::Int(ScalarInt::try_from_uint(offset.bytes(), cx.pointer_size()).unwrap())
            }
        }
    }

    #[inline]
    pub fn null_ptr(cx: &impl HasDataLayout) -> Self {
        Scalar::Int(ScalarInt::null(cx.pointer_size()))
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

    /// This is almost certainly not the method you want!  You should dispatch on the type
    /// and use `to_{u8,u16,...}`/`scalar_to_ptr` to perform ptr-to-int / int-to-ptr casts as needed.
    ///
    /// This method only exists for the benefit of low-level operations that truly need to treat the
    /// scalar in whatever form it is.
    ///
    /// This throws UB (instead of ICEing) on a size mismatch since size mismatches can arise in
    /// Miri when someone declares a function that we shim (such as `malloc`) with a wrong type.
    #[inline]
    pub fn to_bits_or_ptr_internal(
        self,
        target_size: Size,
    ) -> Result<Result<u128, Pointer<Prov>>, ScalarSizeMismatch> {
        assert_ne!(target_size.bytes(), 0, "you should never look at the bits of a ZST");
        Ok(match self {
            Scalar::Int(int) => Ok(int.to_bits(target_size).map_err(|size| {
                ScalarSizeMismatch { target_size: target_size.bytes(), data_size: size.bytes() }
            })?),
            Scalar::Ptr(ptr, sz) => {
                if target_size.bytes() != u64::from(sz) {
                    return Err(ScalarSizeMismatch {
                        target_size: target_size.bytes(),
                        data_size: sz.into(),
                    });
                }
                Err(ptr)
            }
        })
    }
}

impl<'tcx, Prov: Provenance> Scalar<Prov> {
    pub fn to_pointer(self, cx: &impl HasDataLayout) -> InterpResult<'tcx, Pointer<Option<Prov>>> {
        match self
            .to_bits_or_ptr_internal(cx.pointer_size())
            .map_err(|s| err_ub!(ScalarSizeMismatch(s)))?
        {
            Err(ptr) => Ok(ptr.into()),
            Ok(bits) => {
                let addr = u64::try_from(bits).unwrap();
                Ok(Pointer::from_addr(addr))
            }
        }
    }

    /// Fundamental scalar-to-int (cast) operation. Many convenience wrappers exist below, that you
    /// likely want to use instead.
    ///
    /// Will perform ptr-to-int casts if needed and possible.
    /// If that fails, we know the offset is relative, so we return an "erased" Scalar
    /// (which is useful for error messages but not much else).
    #[inline]
    pub fn try_to_int(self) -> Result<ScalarInt, Scalar<AllocId>> {
        match self {
            Scalar::Int(int) => Ok(int),
            Scalar::Ptr(ptr, sz) => {
                if Prov::OFFSET_IS_ADDR {
                    Ok(ScalarInt::try_from_uint(ptr.offset.bytes(), Size::from_bytes(sz)).unwrap())
                } else {
                    // We know `offset` is relative, since `OFFSET_IS_ADDR == false`.
                    let (prov, offset) = ptr.into_parts();
                    // Because `OFFSET_IS_ADDR == false`, this unwrap can never fail.
                    Err(Scalar::Ptr(Pointer::new(prov.get_alloc_id().unwrap(), offset), sz))
                }
            }
        }
    }

    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)] // only in debug builds due to perf (see #98980)
    pub fn assert_int(self) -> ScalarInt {
        self.try_to_int().unwrap()
    }

    /// This throws UB (instead of ICEing) on a size mismatch since size mismatches can arise in
    /// Miri when someone declares a function that we shim (such as `malloc`) with a wrong type.
    #[inline]
    pub fn to_bits(self, target_size: Size) -> InterpResult<'tcx, u128> {
        assert_ne!(target_size.bytes(), 0, "you should never look at the bits of a ZST");
        self.try_to_int().map_err(|_| err_unsup!(ReadPointerAsBytes))?.to_bits(target_size).map_err(
            |size| {
                err_ub!(ScalarSizeMismatch(ScalarSizeMismatch {
                    target_size: target_size.bytes(),
                    data_size: size.bytes(),
                }))
                .into()
            },
        )
    }

    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)] // only in debug builds due to perf (see #98980)
    pub fn assert_bits(self, target_size: Size) -> u128 {
        self.to_bits(target_size).unwrap()
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

    /// Converts the scalar to produce an unsigned integer of the given size.
    /// Fails if the scalar is a pointer.
    #[inline]
    pub fn to_uint(self, size: Size) -> InterpResult<'tcx, u128> {
        self.to_bits(size)
    }

    /// Converts the scalar to produce a `u8`. Fails if the scalar is a pointer.
    pub fn to_u8(self) -> InterpResult<'tcx, u8> {
        self.to_uint(Size::from_bits(8)).map(|v| u8::try_from(v).unwrap())
    }

    /// Converts the scalar to produce a `u16`. Fails if the scalar is a pointer.
    pub fn to_u16(self) -> InterpResult<'tcx, u16> {
        self.to_uint(Size::from_bits(16)).map(|v| u16::try_from(v).unwrap())
    }

    /// Converts the scalar to produce a `u32`. Fails if the scalar is a pointer.
    pub fn to_u32(self) -> InterpResult<'tcx, u32> {
        self.to_uint(Size::from_bits(32)).map(|v| u32::try_from(v).unwrap())
    }

    /// Converts the scalar to produce a `u64`. Fails if the scalar is a pointer.
    pub fn to_u64(self) -> InterpResult<'tcx, u64> {
        self.to_uint(Size::from_bits(64)).map(|v| u64::try_from(v).unwrap())
    }

    /// Converts the scalar to produce a `u128`. Fails if the scalar is a pointer.
    pub fn to_u128(self) -> InterpResult<'tcx, u128> {
        self.to_uint(Size::from_bits(128))
    }

    /// Converts the scalar to produce a machine-pointer-sized unsigned integer.
    /// Fails if the scalar is a pointer.
    pub fn to_machine_usize(self, cx: &impl HasDataLayout) -> InterpResult<'tcx, u64> {
        let b = self.to_uint(cx.data_layout().pointer_size)?;
        Ok(u64::try_from(b).unwrap())
    }

    /// Converts the scalar to produce a signed integer of the given size.
    /// Fails if the scalar is a pointer.
    #[inline]
    pub fn to_int(self, size: Size) -> InterpResult<'tcx, i128> {
        let b = self.to_bits(size)?;
        Ok(size.sign_extend(b) as i128)
    }

    /// Converts the scalar to produce an `i8`. Fails if the scalar is a pointer.
    pub fn to_i8(self) -> InterpResult<'tcx, i8> {
        self.to_int(Size::from_bits(8)).map(|v| i8::try_from(v).unwrap())
    }

    /// Converts the scalar to produce an `i16`. Fails if the scalar is a pointer.
    pub fn to_i16(self) -> InterpResult<'tcx, i16> {
        self.to_int(Size::from_bits(16)).map(|v| i16::try_from(v).unwrap())
    }

    /// Converts the scalar to produce an `i32`. Fails if the scalar is a pointer.
    pub fn to_i32(self) -> InterpResult<'tcx, i32> {
        self.to_int(Size::from_bits(32)).map(|v| i32::try_from(v).unwrap())
    }

    /// Converts the scalar to produce an `i64`. Fails if the scalar is a pointer.
    pub fn to_i64(self) -> InterpResult<'tcx, i64> {
        self.to_int(Size::from_bits(64)).map(|v| i64::try_from(v).unwrap())
    }

    /// Converts the scalar to produce an `i128`. Fails if the scalar is a pointer.
    pub fn to_i128(self) -> InterpResult<'tcx, i128> {
        self.to_int(Size::from_bits(128))
    }

    /// Converts the scalar to produce a machine-pointer-sized signed integer.
    /// Fails if the scalar is a pointer.
    pub fn to_machine_isize(self, cx: &impl HasDataLayout) -> InterpResult<'tcx, i64> {
        let b = self.to_int(cx.data_layout().pointer_size)?;
        Ok(i64::try_from(b).unwrap())
    }

    #[inline]
    pub fn to_f32(self) -> InterpResult<'tcx, Single> {
        // Going through `u32` to check size and truncation.
        Ok(Single::from_bits(self.to_u32()?.into()))
    }

    #[inline]
    pub fn to_f64(self) -> InterpResult<'tcx, Double> {
        // Going through `u64` to check size and truncation.
        Ok(Double::from_bits(self.to_u64()?.into()))
    }
}

/// Gets the bytes of a constant slice value.
pub fn get_slice_bytes<'tcx>(cx: &impl HasDataLayout, val: ConstValue<'tcx>) -> &'tcx [u8] {
    if let ConstValue::Slice { data, start, end } = val {
        let len = end - start;
        data.inner()
            .get_bytes(
                cx,
                AllocRange { start: Size::from_bytes(start), size: Size::from_bytes(len) },
            )
            .unwrap_or_else(|err| bug!("const slice is invalid: {:?}", err))
    } else {
        bug!("expected const slice, but found another const value");
    }
}
