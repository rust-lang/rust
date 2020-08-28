use super::{uabs, AllocId, InterpResult};

use rustc_macros::HashStable;
use rustc_target::abi::{HasDataLayout, Size};

use std::convert::TryFrom;
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
        let max_usize_plus_1 = 1u128 << self.pointer_size().bits();
        u64::try_from(max_usize_plus_1 - 1).unwrap()
    }

    #[inline]
    fn machine_isize_min(&self) -> i64 {
        let max_isize_plus_1 = 1i128 << (self.pointer_size().bits() - 1);
        i64::try_from(-max_isize_plus_1).unwrap()
    }

    #[inline]
    fn machine_isize_max(&self) -> i64 {
        let max_isize_plus_1 = 1u128 << (self.pointer_size().bits() - 1);
        i64::try_from(max_isize_plus_1 - 1).unwrap()
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
        let n = uabs(i);
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

/// Represents a pointer in the Miri engine.
///
/// `Pointer` is generic over the `Tag` associated with each pointer,
/// which is used to do provenance tracking during execution.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, TyEncodable, TyDecodable, Hash)]
#[derive(HashStable)]
pub struct Pointer<Tag = ()> {
    pub alloc_id: AllocId,
    pub offset: Size,
    pub tag: Tag,
}

static_assert_size!(Pointer, 16);

/// Print the address of a pointer (without the tag)
fn print_ptr_addr<Tag>(ptr: &Pointer<Tag>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    // Forward `alternate` flag to `alloc_id` printing.
    if f.alternate() {
        write!(f, "{:#?}", ptr.alloc_id)?;
    } else {
        write!(f, "{:?}", ptr.alloc_id)?;
    }
    // Print offset only if it is non-zero.
    if ptr.offset.bytes() > 0 {
        write!(f, "+0x{:x}", ptr.offset.bytes())?;
    }
    Ok(())
}

// We want the `Debug` output to be readable as it is used by `derive(Debug)` for
// all the Miri types.
// We have to use `Debug` output for the tag, because `()` does not implement
// `Display` so we cannot specialize that.
impl<Tag: fmt::Debug> fmt::Debug for Pointer<Tag> {
    default fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        print_ptr_addr(self, f)?;
        write!(f, "[{:?}]", self.tag)
    }
}
// Specialization for no tag
impl fmt::Debug for Pointer<()> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        print_ptr_addr(self, f)
    }
}

impl<Tag: fmt::Debug> fmt::Display for Pointer<Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Produces a `Pointer` that points to the beginning of the `Allocation`.
impl From<AllocId> for Pointer {
    #[inline(always)]
    fn from(alloc_id: AllocId) -> Self {
        Pointer::new(alloc_id, Size::ZERO)
    }
}

impl Pointer<()> {
    #[inline(always)]
    pub fn new(alloc_id: AllocId, offset: Size) -> Self {
        Pointer { alloc_id, offset, tag: () }
    }

    #[inline(always)]
    pub fn with_tag<Tag>(self, tag: Tag) -> Pointer<Tag> {
        Pointer::new_with_tag(self.alloc_id, self.offset, tag)
    }
}

impl<'tcx, Tag> Pointer<Tag> {
    #[inline(always)]
    pub fn new_with_tag(alloc_id: AllocId, offset: Size, tag: Tag) -> Self {
        Pointer { alloc_id, offset, tag }
    }

    #[inline]
    pub fn offset(self, i: Size, cx: &impl HasDataLayout) -> InterpResult<'tcx, Self> {
        Ok(Pointer::new_with_tag(
            self.alloc_id,
            Size::from_bytes(cx.data_layout().offset(self.offset.bytes(), i.bytes())?),
            self.tag,
        ))
    }

    #[inline]
    pub fn overflowing_offset(self, i: Size, cx: &impl HasDataLayout) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_offset(self.offset.bytes(), i.bytes());
        (Pointer::new_with_tag(self.alloc_id, Size::from_bytes(res), self.tag), over)
    }

    #[inline(always)]
    pub fn wrapping_offset(self, i: Size, cx: &impl HasDataLayout) -> Self {
        self.overflowing_offset(i, cx).0
    }

    #[inline]
    pub fn signed_offset(self, i: i64, cx: &impl HasDataLayout) -> InterpResult<'tcx, Self> {
        Ok(Pointer::new_with_tag(
            self.alloc_id,
            Size::from_bytes(cx.data_layout().signed_offset(self.offset.bytes(), i)?),
            self.tag,
        ))
    }

    #[inline]
    pub fn overflowing_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_signed_offset(self.offset.bytes(), i);
        (Pointer::new_with_tag(self.alloc_id, Size::from_bytes(res), self.tag), over)
    }

    #[inline(always)]
    pub fn wrapping_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> Self {
        self.overflowing_signed_offset(i, cx).0
    }

    #[inline(always)]
    pub fn erase_tag(self) -> Pointer {
        Pointer { alloc_id: self.alloc_id, offset: self.offset, tag: () }
    }
}
