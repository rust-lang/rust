use super::{AllocId, InterpResult};

use rustc_macros::HashStable;
use rustc_target::abi::{HasDataLayout, Size};

use std::fmt;

////////////////////////////////////////////////////////////////////////////////
// Pointer arithmetic
////////////////////////////////////////////////////////////////////////////////

pub trait PointerArithmetic: HasDataLayout {
    // These are not supposed to be overridden.

    #[inline(always)]
    fn pointer_size(&self) -> Size {
        self.data_layout().pointer_size
    }

    #[inline(always)]
    fn max_size_of_val(&self) -> Size {
        Size::from_bytes(self.target_isize_max())
    }

    #[inline]
    fn target_usize_max(&self) -> u64 {
        self.pointer_size().unsigned_int_max().try_into().unwrap()
    }

    #[inline]
    fn target_isize_min(&self) -> i64 {
        self.pointer_size().signed_int_min().try_into().unwrap()
    }

    #[inline]
    fn target_isize_max(&self) -> i64 {
        self.pointer_size().signed_int_max().try_into().unwrap()
    }

    #[inline]
    fn target_usize_to_isize(&self, val: u64) -> i64 {
        let val = val as i64;
        // Now wrap-around into the machine_isize range.
        if val > self.target_isize_max() {
            // This can only happen if the ptr size is < 64, so we know max_usize_plus_1 fits into
            // i64.
            debug_assert!(self.pointer_size().bits() < 64);
            let max_usize_plus_1 = 1u128 << self.pointer_size().bits();
            val - i64::try_from(max_usize_plus_1).unwrap()
        } else {
            val
        }
    }

    /// Helper function: truncate given value-"overflowed flag" pair to pointer size and
    /// update "overflowed flag" if there was an overflow.
    /// This should be called by all the other methods before returning!
    #[inline]
    fn truncate_to_ptr(&self, (val, over): (u64, bool)) -> (u64, bool) {
        let val = u128::from(val);
        let max_ptr_plus_1 = 1u128 << self.pointer_size().bits();
        (u64::try_from(val % max_ptr_plus_1).unwrap(), over || val >= max_ptr_plus_1)
    }

    #[inline]
    fn overflowing_offset(&self, val: u64, i: u64) -> (u64, bool) {
        // We do not need to check if i fits in a machine usize. If it doesn't,
        // either the wrapping_add will wrap or res will not fit in a pointer.
        let res = val.overflowing_add(i);
        self.truncate_to_ptr(res)
    }

    #[inline]
    fn overflowing_signed_offset(&self, val: u64, i: i64) -> (u64, bool) {
        // We need to make sure that i fits in a machine isize.
        let n = i.unsigned_abs();
        if i >= 0 {
            let (val, over) = self.overflowing_offset(val, n);
            (val, over || i > self.target_isize_max())
        } else {
            let res = val.overflowing_sub(n);
            let (val, over) = self.truncate_to_ptr(res);
            (val, over || i < self.target_isize_min())
        }
    }

    #[inline]
    fn offset<'tcx>(&self, val: u64, i: u64) -> InterpResult<'tcx, u64> {
        let (res, over) = self.overflowing_offset(val, i);
        if over { throw_ub!(PointerArithOverflow) } else { Ok(res) }
    }

    #[inline]
    fn signed_offset<'tcx>(&self, val: u64, i: i64) -> InterpResult<'tcx, u64> {
        let (res, over) = self.overflowing_signed_offset(val, i);
        if over { throw_ub!(PointerArithOverflow) } else { Ok(res) }
    }
}

impl<T: HasDataLayout> PointerArithmetic for T {}

/// This trait abstracts over the kind of provenance that is associated with a `Pointer`. It is
/// mostly opaque; the `Machine` trait extends it with some more operations that also have access to
/// some global state.
/// The `Debug` rendering is used to distplay bare provenance, and for the default impl of `fmt`.
pub trait Provenance: Copy + fmt::Debug {
    /// Says whether the `offset` field of `Pointer`s with this provenance is the actual physical address.
    /// - If `false`, the offset *must* be relative. This means the bytes representing a pointer are
    ///   different from what the Abstract Machine prescribes, so the interpreter must prevent any
    ///   operation that would inspect the underlying bytes of a pointer, such as ptr-to-int
    ///   transmutation. A `ReadPointerAsBytes` error will be raised in such situations.
    /// - If `true`, the interpreter will permit operations to inspect the underlying bytes of a
    ///   pointer, and implement ptr-to-int transmutation by stripping provenance.
    const OFFSET_IS_ADDR: bool;

    /// Determines how a pointer should be printed.
    ///
    /// Default impl is only good for when `OFFSET_IS_ADDR == true`.
    fn fmt(ptr: &Pointer<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result
    where
        Self: Sized,
    {
        assert!(Self::OFFSET_IS_ADDR);
        let (prov, addr) = ptr.into_parts(); // address is absolute
        write!(f, "{:#x}", addr.bytes())?;
        if f.alternate() {
            write!(f, "{prov:#?}")?;
        } else {
            write!(f, "{prov:?}")?;
        }
        Ok(())
    }

    /// If `OFFSET_IS_ADDR == false`, provenance must always be able to
    /// identify the allocation this ptr points to (i.e., this must return `Some`).
    /// Otherwise this function is best-effort (but must agree with `Machine::ptr_get_alloc`).
    /// (Identifying the offset in that allocation, however, is harder -- use `Memory::ptr_get_alloc` for that.)
    fn get_alloc_id(self) -> Option<AllocId>;

    /// Defines the 'join' of provenance: what happens when doing a pointer load and different bytes have different provenance.
    fn join(left: Option<Self>, right: Option<Self>) -> Option<Self>;
}

impl Provenance for AllocId {
    // With the `AllocId` as provenance, the `offset` is interpreted *relative to the allocation*,
    // so ptr-to-int casts are not possible (since we do not know the global physical offset).
    const OFFSET_IS_ADDR: bool = false;

    fn fmt(ptr: &Pointer<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Forward `alternate` flag to `alloc_id` printing.
        if f.alternate() {
            write!(f, "{:#?}", ptr.provenance)?;
        } else {
            write!(f, "{:?}", ptr.provenance)?;
        }
        // Print offset only if it is non-zero.
        if ptr.offset.bytes() > 0 {
            write!(f, "+{:#x}", ptr.offset.bytes())?;
        }
        Ok(())
    }

    fn get_alloc_id(self) -> Option<AllocId> {
        Some(self)
    }

    fn join(_left: Option<Self>, _right: Option<Self>) -> Option<Self> {
        panic!("merging provenance is not supported when `OFFSET_IS_ADDR` is false")
    }
}

/// Represents a pointer in the Miri engine.
///
/// Pointers are "tagged" with provenance information; typically the `AllocId` they belong to.
#[derive(Copy, Clone, Eq, PartialEq, TyEncodable, TyDecodable, Hash)]
#[derive(HashStable)]
pub struct Pointer<Prov = AllocId> {
    pub(super) offset: Size, // kept private to avoid accidental misinterpretation (meaning depends on `Prov` type)
    pub provenance: Prov,
}

static_assert_size!(Pointer, 16);
// `Option<Prov>` pointers are also passed around quite a bit
// (but not stored in permanent machine state).
static_assert_size!(Pointer<Option<AllocId>>, 16);

// We want the `Debug` output to be readable as it is used by `derive(Debug)` for
// all the Miri types.
impl<Prov: Provenance> fmt::Debug for Pointer<Prov> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Provenance::fmt(self, f)
    }
}

impl<Prov: Provenance> fmt::Debug for Pointer<Option<Prov>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.provenance {
            Some(prov) => Provenance::fmt(&Pointer::new(prov, self.offset), f),
            None => write!(f, "{:#x}[noalloc]", self.offset.bytes()),
        }
    }
}

impl<Prov: Provenance> fmt::Display for Pointer<Option<Prov>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.provenance.is_none() && self.offset.bytes() == 0 {
            write!(f, "null pointer")
        } else {
            fmt::Debug::fmt(self, f)
        }
    }
}

/// Produces a `Pointer` that points to the beginning of the `Allocation`.
impl From<AllocId> for Pointer {
    #[inline(always)]
    fn from(alloc_id: AllocId) -> Self {
        Pointer::new(alloc_id, Size::ZERO)
    }
}

impl<Prov> From<Pointer<Prov>> for Pointer<Option<Prov>> {
    #[inline(always)]
    fn from(ptr: Pointer<Prov>) -> Self {
        let (prov, offset) = ptr.into_parts();
        Pointer::new(Some(prov), offset)
    }
}

impl<Prov> Pointer<Option<Prov>> {
    /// Convert this pointer that *might* have a provenance into a pointer that *definitely* has a
    /// provenance, or an absolute address.
    ///
    /// This is rarely what you want; call `ptr_try_get_alloc_id` instead.
    pub fn into_pointer_or_addr(self) -> Result<Pointer<Prov>, Size> {
        match self.provenance {
            Some(prov) => Ok(Pointer::new(prov, self.offset)),
            None => Err(self.offset),
        }
    }

    /// Returns the absolute address the pointer points to.
    /// Only works if Prov::OFFSET_IS_ADDR is true!
    pub fn addr(self) -> Size
    where
        Prov: Provenance,
    {
        assert!(Prov::OFFSET_IS_ADDR);
        self.offset
    }
}

impl<Prov> Pointer<Option<Prov>> {
    /// Creates a pointer to the given address, with invalid provenance (i.e., cannot be used for
    /// any memory access).
    #[inline(always)]
    pub fn from_addr_invalid(addr: u64) -> Self {
        Pointer { provenance: None, offset: Size::from_bytes(addr) }
    }

    #[inline(always)]
    pub fn null() -> Self {
        Pointer::from_addr_invalid(0)
    }
}

impl<'tcx, Prov> Pointer<Prov> {
    #[inline(always)]
    pub fn new(provenance: Prov, offset: Size) -> Self {
        Pointer { provenance, offset }
    }

    /// Obtain the constituents of this pointer. Not that the meaning of the offset depends on the type `Prov`!
    /// This function must only be used in the implementation of `Machine::ptr_get_alloc`,
    /// and when a `Pointer` is taken apart to be stored efficiently in an `Allocation`.
    #[inline(always)]
    pub fn into_parts(self) -> (Prov, Size) {
        (self.provenance, self.offset)
    }

    pub fn map_provenance(self, f: impl FnOnce(Prov) -> Prov) -> Self {
        Pointer { provenance: f(self.provenance), ..self }
    }

    #[inline]
    pub fn offset(self, i: Size, cx: &impl HasDataLayout) -> InterpResult<'tcx, Self> {
        Ok(Pointer {
            offset: Size::from_bytes(cx.data_layout().offset(self.offset.bytes(), i.bytes())?),
            ..self
        })
    }

    #[inline]
    pub fn overflowing_offset(self, i: Size, cx: &impl HasDataLayout) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_offset(self.offset.bytes(), i.bytes());
        let ptr = Pointer { offset: Size::from_bytes(res), ..self };
        (ptr, over)
    }

    #[inline(always)]
    pub fn wrapping_offset(self, i: Size, cx: &impl HasDataLayout) -> Self {
        self.overflowing_offset(i, cx).0
    }

    #[inline]
    pub fn signed_offset(self, i: i64, cx: &impl HasDataLayout) -> InterpResult<'tcx, Self> {
        Ok(Pointer {
            offset: Size::from_bytes(cx.data_layout().signed_offset(self.offset.bytes(), i)?),
            ..self
        })
    }

    #[inline]
    pub fn overflowing_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_signed_offset(self.offset.bytes(), i);
        let ptr = Pointer { offset: Size::from_bytes(res), ..self };
        (ptr, over)
    }

    #[inline(always)]
    pub fn wrapping_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> Self {
        self.overflowing_signed_offset(i, cx).0
    }
}
