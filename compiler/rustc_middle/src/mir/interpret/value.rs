use std::fmt;

use either::{Either, Left, Right};
use rustc_abi::{HasDataLayout, Size};
use rustc_apfloat::Float;
use rustc_apfloat::ieee::{Double, Half, Quad, Single};
use rustc_macros::{HashStable, TyDecodable, TyEncodable};

use super::{
    AllocId, CtfeProvenance, InterpResult, Pointer, PointerArithmetic, Provenance,
    ScalarSizeMismatch, interp_ok,
};
use crate::ty::ScalarInt;

/// A `Scalar` represents an immediate, primitive value existing outside of a
/// `memory::Allocation`. It is in many ways like a small chunk of an `Allocation`, up to 16 bytes in
/// size. Like a range of bytes in an `Allocation`, a `Scalar` can either represent the raw bytes
/// of a simple value or a pointer into another `Allocation`
///
/// These variants would be private if there was a convenient way to achieve that in Rust.
/// Do *not* match on a `Scalar`! Use the various `to_*` methods instead.
#[derive(Clone, Copy, Eq, PartialEq, TyEncodable, TyDecodable, Hash)]
#[derive(HashStable)]
pub enum Scalar<Prov = CtfeProvenance> {
    /// The raw bytes of a simple value.
    Int(ScalarInt),

    /// A pointer.
    ///
    /// We also store the size of the pointer, such that a `Scalar` always knows how big it is.
    /// The size is always the pointer size of the current target, but this is not information
    /// that we always have readily available.
    Ptr(Pointer<Prov>, u8),
}

#[cfg(target_pointer_width = "64")]
rustc_data_structures::static_assert_size!(Scalar, 24);

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

impl<Prov> From<Half> for Scalar<Prov> {
    #[inline(always)]
    fn from(f: Half) -> Self {
        Scalar::from_f16(f)
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

impl<Prov> From<Quad> for Scalar<Prov> {
    #[inline(always)]
    fn from(f: Quad) -> Self {
        Scalar::from_f128(f)
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
    pub fn from_uint(i: impl Into<u128>, size: Size) -> Self {
        let i = i.into();
        ScalarInt::try_from_uint(i, size)
            .unwrap_or_else(|| bug!("Unsigned value {:#x} does not fit in {} bits", i, size.bits()))
            .into()
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
    pub fn from_int(i: impl Into<i128>, size: Size) -> Self {
        let i = i.into();
        ScalarInt::try_from_int(i, size)
            .unwrap_or_else(|| bug!("Signed value {:#x} does not fit in {} bits", i, size.bits()))
            .into()
    }

    #[inline]
    pub fn from_i8(i: i8) -> Self {
        Self::Int(i.into())
    }

    #[inline]
    pub fn from_i16(i: i16) -> Self {
        Self::Int(i.into())
    }

    #[inline]
    pub fn from_i32(i: i32) -> Self {
        Self::Int(i.into())
    }

    #[inline]
    pub fn from_i64(i: i64) -> Self {
        Self::Int(i.into())
    }

    #[inline]
    pub fn from_i128(i: i128) -> Self {
        Self::Int(i.into())
    }

    #[inline]
    pub fn from_target_isize(i: i64, cx: &impl HasDataLayout) -> Self {
        Self::from_int(i, cx.data_layout().pointer_size)
    }

    #[inline]
    pub fn from_f16(f: Half) -> Self {
        Scalar::Int(f.into())
    }

    #[inline]
    pub fn from_f32(f: Single) -> Self {
        Scalar::Int(f.into())
    }

    #[inline]
    pub fn from_f64(f: Double) -> Self {
        Scalar::Int(f.into())
    }

    #[inline]
    pub fn from_f128(f: Quad) -> Self {
        Scalar::Int(f.into())
    }

    /// This is almost certainly not the method you want!  You should dispatch on the type
    /// and use `to_{u8,u16,...}`/`to_pointer` to perform ptr-to-int / int-to-ptr casts as needed.
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
            Scalar::Int(int) => Left(int.try_to_bits(target_size).map_err(|size| {
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
            Right(ptr) => interp_ok(ptr.into()),
            Left(bits) => {
                let addr = u64::try_from(bits).unwrap();
                interp_ok(Pointer::from_addr_invalid(addr))
            }
        }
    }

    /// Fundamental scalar-to-int (cast) operation. Many convenience wrappers exist below, that you
    /// likely want to use instead.
    ///
    /// Will perform ptr-to-int casts if needed and possible.
    /// If that fails, we know the offset is relative, so we return an "erased" Scalar
    /// (which is useful for error messages but not much else).
    ///
    /// The error type is `AllocId`, not `CtfeProvenance`, since `AllocId` is the "minimal"
    /// component all provenance types must have.
    #[inline]
    pub fn try_to_scalar_int(self) -> Result<ScalarInt, Scalar<AllocId>> {
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

    pub fn clear_provenance(&mut self) -> InterpResult<'tcx> {
        if matches!(self, Scalar::Ptr(..)) {
            *self = self.to_scalar_int()?.into();
        }
        interp_ok(())
    }

    #[inline(always)]
    pub fn to_scalar_int(self) -> InterpResult<'tcx, ScalarInt> {
        self.try_to_scalar_int().map_err(|_| err_unsup!(ReadPointerAsInt(None))).into()
    }

    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)] // only in debug builds due to perf (see #98980)
    pub fn assert_scalar_int(self) -> ScalarInt {
        self.try_to_scalar_int().expect("got a pointer where a ScalarInt was expected")
    }

    /// This throws UB (instead of ICEing) on a size mismatch since size mismatches can arise in
    /// Miri when someone declares a function that we shim (such as `malloc`) with a wrong type.
    #[inline]
    pub fn to_bits(self, target_size: Size) -> InterpResult<'tcx, u128> {
        assert_ne!(target_size.bytes(), 0, "you should never look at the bits of a ZST");
        self.to_scalar_int()?
            .try_to_bits(target_size)
            .map_err(|size| {
                err_ub!(ScalarSizeMismatch(ScalarSizeMismatch {
                    target_size: target_size.bytes(),
                    data_size: size.bytes(),
                }))
            })
            .into()
    }

    pub fn to_bool(self) -> InterpResult<'tcx, bool> {
        let val = self.to_u8()?;
        match val {
            0 => interp_ok(false),
            1 => interp_ok(true),
            _ => throw_ub!(InvalidBool(val)),
        }
    }

    pub fn to_char(self) -> InterpResult<'tcx, char> {
        let val = self.to_u32()?;
        match std::char::from_u32(val) {
            Some(c) => interp_ok(c),
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
        interp_ok(u64::try_from(b).unwrap())
    }

    /// Converts the scalar to produce a signed integer of the given size.
    /// Fails if the scalar is a pointer.
    #[inline]
    pub fn to_int(self, size: Size) -> InterpResult<'tcx, i128> {
        let b = self.to_bits(size)?;
        interp_ok(size.sign_extend(b))
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
        interp_ok(i64::try_from(b).unwrap())
    }

    #[inline]
    pub fn to_float<F: Float>(self) -> InterpResult<'tcx, F> {
        // Going through `to_bits` to check size and truncation.
        interp_ok(F::from_bits(self.to_bits(Size::from_bits(F::BITS))?))
    }

    #[inline]
    pub fn to_f16(self) -> InterpResult<'tcx, Half> {
        self.to_float()
    }

    #[inline]
    pub fn to_f32(self) -> InterpResult<'tcx, Single> {
        self.to_float()
    }

    #[inline]
    pub fn to_f64(self) -> InterpResult<'tcx, Double> {
        self.to_float()
    }

    #[inline]
    pub fn to_f128(self) -> InterpResult<'tcx, Quad> {
        self.to_float()
    }
}
