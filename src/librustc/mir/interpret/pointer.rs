use mir;
use ty::layout::{self, HasDataLayout, Size};

use super::{
    AllocId, EvalResult, InboundsCheck,
};

////////////////////////////////////////////////////////////////////////////////
// Pointer arithmetic
////////////////////////////////////////////////////////////////////////////////

pub trait PointerArithmetic: layout::HasDataLayout {
    // These are not supposed to be overridden.

    #[inline(always)]
    fn pointer_size(&self) -> Size {
        self.data_layout().pointer_size
    }

    //// Trunace the given value to the pointer size; also return whether there was an overflow
    #[inline]
    fn truncate_to_ptr(&self, val: u128) -> (u64, bool) {
        let max_ptr_plus_1 = 1u128 << self.pointer_size().bits();
        ((val % max_ptr_plus_1) as u64, val >= max_ptr_plus_1)
    }

    #[inline]
    fn offset<'tcx>(&self, val: u64, i: u64) -> EvalResult<'tcx, u64> {
        let (res, over) = self.overflowing_offset(val, i);
        if over { err!(Overflow(mir::BinOp::Add)) } else { Ok(res) }
    }

    #[inline]
    fn overflowing_offset(&self, val: u64, i: u64) -> (u64, bool) {
        let (res, over1) = val.overflowing_add(i);
        let (res, over2) = self.truncate_to_ptr(u128::from(res));
        (res, over1 || over2)
    }

    #[inline]
    fn signed_offset<'tcx>(&self, val: u64, i: i64) -> EvalResult<'tcx, u64> {
        let (res, over) = self.overflowing_signed_offset(val, i128::from(i));
        if over { err!(Overflow(mir::BinOp::Add)) } else { Ok(res) }
    }

    // Overflow checking only works properly on the range from -u64 to +u64.
    #[inline]
    fn overflowing_signed_offset(&self, val: u64, i: i128) -> (u64, bool) {
        // FIXME: is it possible to over/underflow here?
        if i < 0 {
            // trickery to ensure that i64::min_value() works fine
            // this formula only works for true negative values, it panics for zero!
            let n = u64::max_value() - (i as u64) + 1;
            val.overflowing_sub(n)
        } else {
            self.overflowing_offset(val, i as u64)
        }
    }
}

impl<T: layout::HasDataLayout> PointerArithmetic for T {}


/// Pointer is generic over the type that represents a reference to Allocations,
/// thus making it possible for the most convenient representation to be used in
/// each context.
///
/// Defaults to the index based and loosely coupled AllocId.
///
/// Pointer is also generic over the `Tag` associated with each pointer,
/// which is used to do provenance tracking during execution.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, RustcEncodable, RustcDecodable, Hash)]
pub struct Pointer<Tag=(),Id=AllocId> {
    pub alloc_id: Id,
    pub offset: Size,
    pub tag: Tag,
}

static_assert!(POINTER_SIZE: ::std::mem::size_of::<Pointer>() == 16);

/// Produces a `Pointer` which points to the beginning of the Allocation
impl From<AllocId> for Pointer {
    #[inline(always)]
    fn from(alloc_id: AllocId) -> Self {
        Pointer::new(alloc_id, Size::ZERO)
    }
}

impl<'tcx> Pointer<()> {
    #[inline(always)]
    pub fn new(alloc_id: AllocId, offset: Size) -> Self {
        Pointer { alloc_id, offset, tag: () }
    }

    #[inline(always)]
    pub fn with_default_tag<Tag>(self) -> Pointer<Tag>
        where Tag: Default
    {
        Pointer::new_with_tag(self.alloc_id, self.offset, Default::default())
    }
}

impl<'tcx, Tag> Pointer<Tag> {
    #[inline(always)]
    pub fn new_with_tag(alloc_id: AllocId, offset: Size, tag: Tag) -> Self {
        Pointer { alloc_id, offset, tag }
    }

    #[inline]
    pub fn offset(self, i: Size, cx: &impl HasDataLayout) -> EvalResult<'tcx, Self> {
        Ok(Pointer::new_with_tag(
            self.alloc_id,
            Size::from_bytes(cx.data_layout().offset(self.offset.bytes(), i.bytes())?),
            self.tag
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
    pub fn signed_offset(self, i: i64, cx: &impl HasDataLayout) -> EvalResult<'tcx, Self> {
        Ok(Pointer::new_with_tag(
            self.alloc_id,
            Size::from_bytes(cx.data_layout().signed_offset(self.offset.bytes(), i)?),
            self.tag,
        ))
    }

    #[inline]
    pub fn overflowing_signed_offset(self, i: i128, cx: &impl HasDataLayout) -> (Self, bool) {
        let (res, over) = cx.data_layout().overflowing_signed_offset(self.offset.bytes(), i);
        (Pointer::new_with_tag(self.alloc_id, Size::from_bytes(res), self.tag), over)
    }

    #[inline(always)]
    pub fn wrapping_signed_offset(self, i: i64, cx: &impl HasDataLayout) -> Self {
        self.overflowing_signed_offset(i128::from(i), cx).0
    }

    #[inline(always)]
    pub fn erase_tag(self) -> Pointer {
        Pointer { alloc_id: self.alloc_id, offset: self.offset, tag: () }
    }

    #[inline(always)]
    pub fn check_in_alloc(
        self,
        allocation_size: Size,
        check: InboundsCheck,
    ) -> EvalResult<'tcx, ()> {
        if self.offset > allocation_size {
            err!(PointerOutOfBounds {
                ptr: self.erase_tag(),
                check,
                allocation_size,
            })
        } else {
            Ok(())
        }
    }
}
