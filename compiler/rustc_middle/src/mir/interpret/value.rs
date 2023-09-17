use std::fmt;

use either::{Either, Left, Right};

use rustc_apfloat::{
    ieee::{Double, Single},
    Float,
};
use rustc_data_structures::intern::Interned;
use rustc_macros::HashStable;
use rustc_target::abi::{HasDataLayout, Size};

use crate::ty::{ParamEnv, ScalarInt, Ty, TyCtxt};

use super::{
    alloc_range, AllocId, InterpResult, Pointer, PointerArithmetic, Provenance, ScalarSizeMismatch,
};

/// Represents the result of const evaluation via the `eval_to_allocation` query.
#[derive(Copy, Clone, HashStable, TyEncodable, TyDecodable, Debug, Hash, Eq, PartialEq)]
pub struct ConstAlloc<'tcx> {
    /// The value lives here, at offset 0, and that allocation definitely is an `AllocKind::Memory`
    /// (so you can use `AllocMap::unwrap_memory`).
    pub alloc_id: AllocId,
    pub ty: Ty<'tcx>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[derive(HashStable)]
pub struct ConstValue<'tcx>(pub(crate) Interned<'tcx, ConstValueKind>);

/// Represents a constant value in Rust. `Scalar` and `Slice` are optimizations for
/// array length computations, enum discriminants and the pattern matching logic.
#[derive(Copy, Clone, Debug, Eq, PartialEq, TyEncodable, TyDecodable, Hash)]
#[derive(HashStable)]
pub enum ConstValueKind {
    /// Used for types with `layout::abi::Scalar` ABI.
    ///
    /// Not using the enum `Value` to encode that this must not be `Uninit`.
    Scalar(Scalar),

    /// Used for types with `layout::abi::ScalarPair` ABI.
    ///
    /// Not using the enum `Value` to encode that this must not be `Uninit`.
    ScalarPair(Scalar, Scalar),

    /// Only for ZSTs.
    ZeroSized,

    /// A value not representable by the other variants; needs to be stored in-memory.
    ///
    /// Must *not* be used for scalars or ZST, but having `&str` or other slices in this variant is fine.
    Indirect {
        /// The backing memory of the value. May contain more memory than needed for just the value
        /// if this points into some other larger ConstValue.
        ///
        /// We use an `AllocId` here instead of a `ConstAllocation<'tcx>` to make sure that when a
        /// raw constant (which is basically just an `AllocId`) is turned into a `ConstValue` and
        /// back, we can preserve the original `AllocId`.
        alloc_id: AllocId,
        /// Offset into `alloc`
        offset: Size,
    },
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(ConstValue<'_>, 8);

impl<'tcx> ConstValue<'tcx> {
    #[inline]
    pub fn new(tcx: TyCtxt<'tcx>, kind: ConstValueKind) -> ConstValue<'tcx> {
        tcx.intern_const_value(kind)
    }

    #[inline]
    pub fn kind(self) -> &'tcx ConstValueKind {
        self.0.0
    }

    #[inline]
    pub fn try_to_scalar(self) -> Option<Scalar<AllocId>> {
        match self.kind() {
            ConstValueKind::Indirect { .. }
            | ConstValueKind::ScalarPair(..)
            | ConstValueKind::ZeroSized => None,
            ConstValueKind::Scalar(val) => Some(*val),
        }
    }

    pub fn try_to_scalar_int(self) -> Option<ScalarInt> {
        self.try_to_scalar()?.try_to_int().ok()
    }

    pub fn try_to_bits(self, size: Size) -> Option<u128> {
        self.try_to_scalar_int()?.to_bits(size).ok()
    }

    pub fn try_to_bool(self) -> Option<bool> {
        self.try_to_scalar_int()?.try_into().ok()
    }

    pub fn try_to_target_usize(self, tcx: TyCtxt<'tcx>) -> Option<u64> {
        self.try_to_scalar_int()?.try_to_target_usize(tcx).ok()
    }

    pub fn try_to_bits_for_ty(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ParamEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<u128> {
        let size = tcx.layout_of(param_env.with_reveal_all_normalized(tcx).and(ty)).ok()?.size;
        self.try_to_bits(size)
    }

    #[inline]
    pub fn zero_sized(tcx: TyCtxt<'tcx>) -> Self {
        Self::new(tcx, ConstValueKind::ZeroSized)
    }

    #[inline]
    pub fn from_scalar(tcx: TyCtxt<'tcx>, scalar: Scalar) -> Self {
        Self::new(tcx, ConstValueKind::Scalar(scalar))
    }

    #[inline]
    pub fn from_bool(tcx: TyCtxt<'tcx>, b: bool) -> Self {
        Self::from_scalar(tcx, Scalar::from_bool(b))
    }

    #[inline]
    pub fn from_u64(tcx: TyCtxt<'tcx>, i: u64) -> Self {
        Self::from_scalar(tcx, Scalar::from_u64(i))
    }

    #[inline]
    pub fn from_u128(tcx: TyCtxt<'tcx>, i: u128) -> Self {
        Self::from_scalar(tcx, Scalar::from_u128(i))
    }

    #[inline]
    pub fn from_target_usize(tcx: TyCtxt<'tcx>, i: u64) -> Self {
        Self::from_scalar(tcx, Scalar::from_target_usize(i, &tcx))
    }

    #[inline]
    pub fn from_pointer(tcx: TyCtxt<'tcx>, pointer: Pointer) -> Self {
        Self::from_scalar(tcx, Scalar::from_pointer(pointer, &tcx))
    }

    #[inline]
    pub fn from_pair(tcx: TyCtxt<'tcx>, a: Scalar, b: Scalar) -> Self {
        Self::new(tcx, ConstValueKind::ScalarPair(a, b))
    }

    #[inline]
    pub fn from_slice(tcx: TyCtxt<'tcx>, pointer: Pointer, length: usize) -> Self {
        Self::from_pair(
            tcx,
            Scalar::from_pointer(pointer, &tcx),
            Scalar::from_target_usize(length as u64, &tcx),
        )
    }

    #[inline]
    pub fn from_memory(tcx: TyCtxt<'tcx>, alloc_id: AllocId, offset: Size) -> Self {
        debug_assert!(matches!(tcx.global_alloc(alloc_id), super::GlobalAlloc::Memory(_)));
        Self::new(tcx, ConstValueKind::Indirect { alloc_id, offset })
    }

    /// Must only be called on constants of type `&str` or `&[u8]`!
    pub fn try_get_slice_bytes_for_diagnostics(self, tcx: TyCtxt<'tcx>) -> Option<&'tcx [u8]> {
        let (data, start, end) = match self.kind() {
            &ConstValueKind::ScalarPair(Scalar::Ptr(pointer, _), Scalar::Int(length)) => {
                let (alloc_id, start) = pointer.into_parts();
                let alloc = tcx.global_alloc(alloc_id).unwrap_memory();
                let start = start.bytes_usize();
                let length = length.try_to_target_usize(tcx).unwrap() as usize;
                (alloc, start, start + length)
            }
            &ConstValueKind::Indirect { alloc_id, offset } => {
                // The reference itself is stored behind an indirection.
                // Load the reference, and then load the actual slice contents.
                let a = tcx.global_alloc(alloc_id).unwrap_memory().inner();
                let ptr_size = tcx.data_layout.pointer_size;
                if a.size() < offset + 2 * ptr_size {
                    // (partially) dangling reference
                    return None;
                }
                // Read the wide pointer components.
                let ptr = a
                    .read_scalar(
                        &tcx,
                        alloc_range(offset, ptr_size),
                        /* read_provenance */ true,
                    )
                    .ok()?;
                let ptr = ptr.to_pointer(&tcx).ok()?;
                let len = a
                    .read_scalar(
                        &tcx,
                        alloc_range(offset + ptr_size, ptr_size),
                        /* read_provenance */ false,
                    )
                    .ok()?;
                let len = len.to_target_usize(&tcx).ok()?;
                let len: usize = len.try_into().ok()?;
                if len == 0 {
                    return Some(&[]);
                }
                // Non-empty slice, must have memory. We know this is a relative pointer.
                let (inner_alloc_id, offset) = ptr.into_parts();
                let data = tcx.global_alloc(inner_alloc_id?).unwrap_memory();
                (data, offset.bytes_usize(), offset.bytes_usize() + len)
            }
            _ => {
                bug!("`try_get_slice_bytes` on non-slice constant")
            }
        };

        // This is for diagnostics only, so we are okay to use `inspect_with_uninit_and_ptr_outside_interpreter`.
        Some(data.inner().inspect_with_uninit_and_ptr_outside_interpreter(start..end))
    }
}

/// A `Scalar` represents an immediate, primitive value existing outside of a
/// `memory::Allocation`. It is in many ways like a small chunk of an `Allocation`, up to 16 bytes in
/// size. Like a range of bytes in an `Allocation`, a `Scalar` can either represent the raw bytes
/// of a simple value or a pointer into another `Allocation`
///
/// These variants would be private if there was a convenient way to achieve that in Rust.
/// Do *not* match on a `Scalar`! Use the various `to_*` methods instead.
#[derive(Clone, Copy, Eq, PartialEq, TyEncodable, TyDecodable, Hash)]
#[derive(HashStable)]
pub enum Scalar<Prov = AllocId> {
    /// The raw bytes of a simple value.
    Int(ScalarInt),

    /// A pointer.
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
            Scalar::Ptr(ptr, _size) => write!(f, "{ptr:?}"),
            Scalar::Int(int) => write!(f, "{int:?}"),
        }
    }
}

impl<Prov: Provenance> fmt::Display for Scalar<Prov> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Ptr(ptr, _size) => write!(f, "pointer to {ptr:?}"),
            Scalar::Int(int) => write!(f, "{int}"),
        }
    }
}

impl<Prov: Provenance> fmt::LowerHex for Scalar<Prov> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Ptr(ptr, _size) => write!(f, "pointer to {ptr:?}"),
            Scalar::Int(int) => write!(f, "{int:#x}"),
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
    pub fn from_u128(i: u128) -> Self {
        Scalar::Int(i.into())
    }

    #[inline]
    pub fn from_target_usize(i: u64, cx: &impl HasDataLayout) -> Self {
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
    pub fn from_target_isize(i: i64, cx: &impl HasDataLayout) -> Self {
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
    ) -> Result<Either<u128, Pointer<Prov>>, ScalarSizeMismatch> {
        assert_ne!(target_size.bytes(), 0, "you should never look at the bits of a ZST");
        Ok(match self {
            Scalar::Int(int) => Left(int.to_bits(target_size).map_err(|size| {
                ScalarSizeMismatch { target_size: target_size.bytes(), data_size: size.bytes() }
            })?),
            Scalar::Ptr(ptr, sz) => {
                if target_size.bytes() != u64::from(sz) {
                    return Err(ScalarSizeMismatch {
                        target_size: target_size.bytes(),
                        data_size: sz.into(),
                    });
                }
                Right(ptr)
            }
        })
    }

    #[inline]
    pub fn size(self) -> Size {
        match self {
            Scalar::Int(int) => int.size(),
            Scalar::Ptr(_ptr, sz) => Size::from_bytes(sz),
        }
    }
}

impl<'tcx, Prov: Provenance> Scalar<Prov> {
    pub fn to_pointer(self, cx: &impl HasDataLayout) -> InterpResult<'tcx, Pointer<Option<Prov>>> {
        match self
            .to_bits_or_ptr_internal(cx.pointer_size())
            .map_err(|s| err_ub!(ScalarSizeMismatch(s)))?
        {
            Right(ptr) => Ok(ptr.into()),
            Left(bits) => {
                let addr = u64::try_from(bits).unwrap();
                Ok(Pointer::from_addr_invalid(addr))
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
        self.try_to_int()
            .map_err(|_| err_unsup!(ReadPointerAsInt(None)))?
            .to_bits(target_size)
            .map_err(|size| {
                err_ub!(ScalarSizeMismatch(ScalarSizeMismatch {
                    target_size: target_size.bytes(),
                    data_size: size.bytes(),
                }))
                .into()
            })
    }

    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)] // only in debug builds due to perf (see #98980)
    pub fn assert_bits(self, target_size: Size) -> u128 {
        self.to_bits(target_size)
            .unwrap_or_else(|_| panic!("assertion failed: {self:?} fits {target_size:?}"))
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
    pub fn to_target_usize(self, cx: &impl HasDataLayout) -> InterpResult<'tcx, u64> {
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
    pub fn to_target_isize(self, cx: &impl HasDataLayout) -> InterpResult<'tcx, i64> {
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
