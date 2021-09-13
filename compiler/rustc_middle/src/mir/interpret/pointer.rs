use super::{AllocId, InterpResult};

use rustc_macros::HashStable;
use rustc_target::abi::{HasDataLayout, Size};

use std::convert::{TryFrom, TryInto};
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

    #[inline]
    fn machine_usize_max(&self) -> u64 {
        self.pointer_size().unsigned_int_max().try_into().unwrap()
    }

    #[inline]
    fn machine_isize_min(&self) -> i64 {
        self.pointer_size().signed_int_min().try_into().unwrap()
    }

    #[inline]
    fn machine_isize_max(&self) -> i64 {
        self.pointer_size().signed_int_max().try_into().unwrap()
    }

    #[inline]
    fn machine_usize_to_isize(&self, val: u64) -> i64 {
        let val = val as i64;
        // Now wrap-around into the machine_isize range.
        if val > self.machine_isize_max() {
            // This can only happen the the ptr size is < 64, so we know max_usize_plus_1 fits into
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
            (val, over || i > self.machine_isize_max())
        } else {
            let res = val.overflowing_sub(n);
            let (val, over) = self.truncate_to_ptr(res);
            (val, over || i < self.machine_isize_min())
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
/// We don't actually care about this `Debug` bound (we use `Provenance::fmt` to format the entire
/// pointer), but `derive` adds some unecessary bounds.
pub trait Provenance: Copy + fmt::Debug {
    /// Says whether the `offset` field of `Pointer`s with this provenance is the actual physical address.
    /// If `true, ptr-to-int casts work by simply discarding the provenance.
    /// If `false`, ptr-to-int casts are not supported. The offset *must* be relative in that case.
    const OFFSET_IS_ADDR: bool;

    /// We also use this trait to control whether to abort execution when a pointer is being partially overwritten
    /// (this avoids a separate trait in `allocation.rs` just for this purpose).
    const ERR_ON_PARTIAL_PTR_OVERWRITE: bool;

    /// Determines how a pointer should be printed.
    fn fmt(ptr: &Pointer<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result
    where
        Self: Sized;

    /// Provenance must always be able to identify the allocation this ptr points to.
    /// (Identifying the offset in that allocation, however, is harder -- use `Memory::ptr_get_alloc` for that.)
    fn get_alloc_id(self) -> AllocId;
}

impl Provenance for AllocId {
    // With the `AllocId` as provenance, the `offset` is interpreted *relative to the allocation*,
    // so ptr-to-int casts are not possible (since we do not know the global physical offset).
    const OFFSET_IS_ADDR: bool = false;

    // For now, do not allow this, so that we keep our options open.
    const ERR_ON_PARTIAL_PTR_OVERWRITE: bool = true;

    fn fmt(ptr: &Pointer<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Forward `alternate` flag to `alloc_id` printing.
        if f.alternate() {
            write!(f, "{:#?}", ptr.provenance)?;
        } else {
            write!(f, "{:?}", ptr.provenance)?;
        }
        // Print offset only if it is non-zero.
        if ptr.offset.bytes() > 0 {
            write!(f, "+0x{:x}", ptr.offset.bytes())?;
        }
        Ok(())
    }

    fn get_alloc_id(self) -> AllocId {
        self
    }
}

/// Represents a pointer in the Miri engine.
///
/// Pointers are "tagged" with provenance information; typically the `AllocId` they belong to.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, TyEncodable, TyDecodable, Hash)]
#[derive(HashStable)]
pub struct Pointer<Tag = AllocId> {
    pub(super) offset: Size, // kept private to avoid accidental misinterpretation (meaning depends on `Tag` type)
    pub provenance: Tag,
}

static_assert_size!(Pointer, 16);

// We want the `Debug` output to be readable as it is used by `derive(Debug)` for
// all the Miri types.
impl<Tag: Provenance> fmt::Debug for Pointer<Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Provenance::fmt(self, f)
    }
}

impl<Tag: Provenance> fmt::Debug for Pointer<Option<Tag>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.provenance {
            Some(tag) => Provenance::fmt(&Pointer::new(tag, self.offset), f),
            None => write!(f, "0x{:x}", self.offset.bytes()),
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

impl<Tag> From<Pointer<Tag>> for Pointer<Option<Tag>> {
    #[inline(always)]
    fn from(ptr: Pointer<Tag>) -> Self {
        let (tag, offset) = ptr.into_parts();
        Pointer::new(Some(tag), offset)
    }
}

impl<Tag> Pointer<Option<Tag>> {
    pub fn into_pointer_or_addr(self) -> Result<Pointer<Tag>, Size> {
        match self.provenance {
            Some(tag) => Ok(Pointer::new(tag, self.offset)),
            None => Err(self.offset),
        }
    }
}

impl<Tag> Pointer<Option<Tag>> {
    #[inline(always)]
    pub fn null() -> Self {
        Pointer { provenance: None, offset: Size::ZERO }
    }
}

impl<'tcx, Tag> Pointer<Tag> {
    #[inline(always)]
    pub fn new(provenance: Tag, offset: Size) -> Self {
        Pointer { provenance, offset }
    }

    /// Obtain the constituents of this pointer. Not that the meaning of the offset depends on the type `Tag`!
    /// This function must only be used in the implementation of `Machine::ptr_get_alloc`,
    /// and when a `Pointer` is taken apart to be stored efficiently in an `Allocation`.
    #[inline(always)]
    pub fn into_parts(self) -> (Tag, Size) {
        (self.provenance, self.offset)
    }

    pub fn map_provenance(self, f: impl FnOnce(Tag) -> Tag) -> Self {
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
