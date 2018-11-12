// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The virtual memory representation of the MIR interpreter

use super::{Pointer, EvalResult, AllocId};

use ty::layout::{Size, Align};
use syntax::ast::Mutability;
use std::iter;
use mir;
use std::ops::{Deref, DerefMut};
use rustc_data_structures::sorted_map::SortedMap;

/// Used by `check_bounds` to indicate whether the pointer needs to be just inbounds
/// or also inbounds of a *live* allocation.
#[derive(Debug, Copy, Clone, RustcEncodable, RustcDecodable)]
pub enum InboundsCheck {
    Live,
    MaybeDead,
}

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable)]
pub struct Allocation<Tag=(),Extra=()> {
    /// The actual bytes of the allocation.
    /// Note that the bytes of a pointer represent the offset of the pointer
    pub bytes: Vec<u8>,
    /// Maps from byte addresses to extra data for each pointer.
    /// Only the first byte of a pointer is inserted into the map; i.e.,
    /// every entry in this map applies to `pointer_size` consecutive bytes starting
    /// at the given offset.
    pub relocations: Relocations<Tag>,
    /// Denotes undefined memory. Reading from undefined memory is forbidden in miri
    pub undef_mask: UndefMask,
    /// The alignment of the allocation to detect unaligned reads.
    pub align: Align,
    /// Whether the allocation is mutable.
    /// Also used by codegen to determine if a static should be put into mutable memory,
    /// which happens for `static mut` and `static` with interior mutability.
    pub mutability: Mutability,
    /// Extra state for the machine.
    pub extra: Extra,
}

/// Byte accessors
impl<'tcx, Tag, Extra> Allocation<Tag, Extra> {
    /// The last argument controls whether we error out when there are undefined
    /// or pointer bytes.  You should never call this, call `get_bytes` or
    /// `get_bytes_with_undef_and_ptr` instead,
    ///
    /// This function also guarantees that the resulting pointer will remain stable
    /// even when new allocations are pushed to the `HashMap`. `copy_repeatedly` relies
    /// on that.
    fn get_bytes_internal(
        &self,
        ptr: Pointer<Tag>,
        size: Size,
        align: Align,
        check_defined_and_ptr: bool,
    ) -> EvalResult<'tcx, &[u8]> {
        assert_ne!(size.bytes(), 0, "0-sized accesses should never even get a `Pointer`");
        self.check_align(ptr.into(), align)?;
        self.check_bounds(ptr, size, InboundsCheck::Live)?;

        if check_defined_and_ptr {
            self.check_defined(ptr, size)?;
            self.check_relocations(ptr, size)?;
        } else {
            // We still don't want relocations on the *edges*
            self.check_relocation_edges(ptr, size)?;
        }

        let alloc = self.get(ptr.alloc_id)?;
        AllocationExtra::memory_read(alloc, ptr, size)?;

        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        let offset = ptr.offset.bytes() as usize;
        Ok(&alloc.bytes[offset..offset + size.bytes() as usize])
    }

    #[inline]
    fn get_bytes(
        &self,
        ptr: Pointer<Tag>,
        size: Size,
        align: Align
    ) -> EvalResult<'tcx, &[u8]> {
        self.get_bytes_internal(ptr, size, align, true)
    }

    /// It is the caller's responsibility to handle undefined and pointer bytes.
    /// However, this still checks that there are no relocations on the *edges*.
    #[inline]
    fn get_bytes_with_undef_and_ptr(
        &self,
        ptr: Pointer<Tag>,
        size: Size,
        align: Align
    ) -> EvalResult<'tcx, &[u8]> {
        self.get_bytes_internal(ptr, size, align, false)
    }

    /// Just calling this already marks everything as defined and removes relocations,
    /// so be sure to actually put data there!
    fn get_bytes_mut(
        &mut self,
        ptr: Pointer<Tag>,
        size: Size,
        align: Align,
    ) -> EvalResult<'tcx, &mut [u8]> {
        assert_ne!(size.bytes(), 0, "0-sized accesses should never even get a `Pointer`");
        self.check_align(ptr.into(), align)?;
        self.check_bounds(ptr, size, InboundsCheck::Live)?;

        self.mark_definedness(ptr, size, true)?;
        self.clear_relocations(ptr, size)?;

        let alloc = self.get_mut(ptr.alloc_id)?;
        AllocationExtra::memory_written(alloc, ptr, size)?;

        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        let offset = ptr.offset.bytes() as usize;
        Ok(&mut alloc.bytes[offset..offset + size.bytes() as usize])
    }
}

/// Relocations
impl<'tcx, Tag, Extra> Allocation<Tag, Extra> {
    /// Return all relocations overlapping with the given ptr-offset pair.
    fn relocations(
        &self,
        ptr: Pointer<M::PointerTag>,
        size: Size,
    ) -> EvalResult<'tcx, &[(Size, (M::PointerTag, AllocId))]> {
        // We have to go back `pointer_size - 1` bytes, as that one would still overlap with
        // the beginning of this range.
        let start = ptr.offset.bytes().saturating_sub(self.pointer_size().bytes() - 1);
        let end = ptr.offset + size; // this does overflow checking
        Ok(self.get(ptr.alloc_id)?.relocations.range(Size::from_bytes(start)..end))
    }

    /// Check that there ar eno relocations overlapping with the given range.
    #[inline(always)]
    fn check_relocations(&self, ptr: Pointer<M::PointerTag>, size: Size) -> EvalResult<'tcx> {
        if self.relocations(ptr, size)?.len() != 0 {
            err!(ReadPointerAsBytes)
        } else {
            Ok(())
        }
    }

    /// Remove all relocations inside the given range.
    /// If there are relocations overlapping with the edges, they
    /// are removed as well *and* the bytes they cover are marked as
    /// uninitialized.  This is a somewhat odd "spooky action at a distance",
    /// but it allows strictly more code to run than if we would just error
    /// immediately in that case.
    fn clear_relocations(&mut self, ptr: Pointer<M::PointerTag>, size: Size) -> EvalResult<'tcx> {
        // Find the start and end of the given range and its outermost relocations.
        let (first, last) = {
            // Find all relocations overlapping the given range.
            let relocations = self.relocations(ptr, size)?;
            if relocations.is_empty() {
                return Ok(());
            }

            (relocations.first().unwrap().0,
             relocations.last().unwrap().0 + self.pointer_size())
        };
        let start = ptr.offset;
        let end = start + size;

        let alloc = self.get_mut(ptr.alloc_id)?;

        // Mark parts of the outermost relocations as undefined if they partially fall outside the
        // given range.
        if first < start {
            alloc.undef_mask.set_range(first, start, false);
        }
        if last > end {
            alloc.undef_mask.set_range(end, last, false);
        }

        // Forget all the relocations.
        alloc.relocations.remove_range(first..last);

        Ok(())
    }

    /// Error if there are relocations overlapping with the edges of the
    /// given memory range.
    #[inline]
    fn check_relocation_edges(&self, ptr: Pointer<M::PointerTag>, size: Size) -> EvalResult<'tcx> {
        self.check_relocations(ptr, Size::ZERO)?;
        self.check_relocations(ptr.offset(size, self)?, Size::ZERO)?;
        Ok(())
    }
}


/// Undefined bytes
impl<'tcx, Tag, Extra> Allocation<Tag, Extra> {
    /// Checks that a range of bytes is defined. If not, returns the `ReadUndefBytes`
    /// error which will report the first byte which is undefined.
    #[inline]
    fn check_defined(&self, ptr: Pointer<M::PointerTag>, size: Size) -> EvalResult<'tcx> {
        let alloc = self.get(ptr.alloc_id)?;
        alloc.undef_mask.is_range_defined(
            ptr.offset,
            ptr.offset + size,
        ).or_else(|idx| err!(ReadUndefBytes(idx)))
    }

    pub fn mark_definedness(
        &mut self,
        ptr: Pointer<M::PointerTag>,
        size: Size,
        new_state: bool,
    ) -> EvalResult<'tcx> {
        if size.bytes() == 0 {
            return Ok(());
        }
        let alloc = self.get_mut(ptr.alloc_id)?;
        alloc.undef_mask.set_range(
            ptr.offset,
            ptr.offset + size,
            new_state,
        );
        Ok(())
    }
}

pub trait AllocationExtra<Tag>: ::std::fmt::Debug + Default + Clone {
    /// Hook for performing extra checks on a memory read access.
    ///
    /// Takes read-only access to the allocation so we can keep all the memory read
    /// operations take `&self`.  Use a `RefCell` in `AllocExtra` if you
    /// need to mutate.
    #[inline]
    fn memory_read(
        _alloc: &Allocation<Tag, Self>,
        _ptr: Pointer<Tag>,
        _size: Size,
    ) -> EvalResult<'tcx> {
        Ok(())
    }

    /// Hook for performing extra checks on a memory write access.
    #[inline]
    fn memory_written(
        _alloc: &mut Allocation<Tag, Self>,
        _ptr: Pointer<Tag>,
        _size: Size,
    ) -> EvalResult<'tcx> {
        Ok(())
    }

    /// Hook for performing extra checks on a memory deallocation.
    /// `size` will be the size of the allocation.
    #[inline]
    fn memory_deallocated(
        _alloc: &mut Allocation<Tag, Self>,
        _ptr: Pointer<Tag>,
        _size: Size,
    ) -> EvalResult<'tcx> {
        Ok(())
    }
}

impl AllocationExtra<()> for () {}

impl<Tag, Extra: Default> Allocation<Tag, Extra> {
    /// Creates a read-only allocation initialized by the given bytes
    pub fn from_bytes(slice: &[u8], align: Align) -> Self {
        let mut undef_mask = UndefMask::new(Size::ZERO);
        undef_mask.grow(Size::from_bytes(slice.len() as u64), true);
        Self {
            bytes: slice.to_owned(),
            relocations: Relocations::new(),
            undef_mask,
            align,
            mutability: Mutability::Immutable,
            extra: Extra::default(),
        }
    }

    pub fn from_byte_aligned_bytes(slice: &[u8]) -> Self {
        Allocation::from_bytes(slice, Align::from_bytes(1).unwrap())
    }

    pub fn undef(size: Size, align: Align) -> Self {
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        Allocation {
            bytes: vec![0; size.bytes() as usize],
            relocations: Relocations::new(),
            undef_mask: UndefMask::new(size),
            align,
            mutability: Mutability::Mutable,
            extra: Extra::default(),
        }
    }
}

impl<'tcx> ::serialize::UseSpecializedDecodable for &'tcx Allocation {}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct Relocations<Tag=(), Id=AllocId>(SortedMap<Size, (Tag, Id)>);

impl<Tag, Id> Relocations<Tag, Id> {
    pub fn new() -> Self {
        Relocations(SortedMap::new())
    }

    // The caller must guarantee that the given relocations are already sorted
    // by address and contain no duplicates.
    pub fn from_presorted(r: Vec<(Size, (Tag, Id))>) -> Self {
        Relocations(SortedMap::from_presorted_elements(r))
    }
}

impl<Tag> Deref for Relocations<Tag> {
    type Target = SortedMap<Size, (Tag, AllocId)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Tag> DerefMut for Relocations<Tag> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

////////////////////////////////////////////////////////////////////////////////
// Undefined byte tracking
////////////////////////////////////////////////////////////////////////////////

type Block = u64;
const BLOCK_SIZE: u64 = 64;

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable)]
pub struct UndefMask {
    blocks: Vec<Block>,
    len: Size,
}

impl_stable_hash_for!(struct mir::interpret::UndefMask{blocks, len});

impl UndefMask {
    pub fn new(size: Size) -> Self {
        let mut m = UndefMask {
            blocks: vec![],
            len: Size::ZERO,
        };
        m.grow(size, false);
        m
    }

    /// Check whether the range `start..end` (end-exclusive) is entirely defined.
    ///
    /// Returns `Ok(())` if it's defined. Otherwise returns the index of the byte
    /// at which the first undefined access begins.
    #[inline]
    pub fn is_range_defined(&self, start: Size, end: Size) -> Result<(), Size> {
        if end > self.len {
            return Err(self.len);
        }

        let idx = (start.bytes()..end.bytes())
            .map(|i| Size::from_bytes(i))
            .find(|&i| !self.get(i));

        match idx {
            Some(idx) => Err(idx),
            None => Ok(())
        }
    }

    pub fn set_range(&mut self, start: Size, end: Size, new_state: bool) {
        let len = self.len;
        if end > len {
            self.grow(end - len, new_state);
        }
        self.set_range_inbounds(start, end, new_state);
    }

    pub fn set_range_inbounds(&mut self, start: Size, end: Size, new_state: bool) {
        for i in start.bytes()..end.bytes() {
            self.set(Size::from_bytes(i), new_state);
        }
    }

    #[inline]
    pub fn get(&self, i: Size) -> bool {
        let (block, bit) = bit_index(i);
        (self.blocks[block] & 1 << bit) != 0
    }

    #[inline]
    pub fn set(&mut self, i: Size, new_state: bool) {
        let (block, bit) = bit_index(i);
        if new_state {
            self.blocks[block] |= 1 << bit;
        } else {
            self.blocks[block] &= !(1 << bit);
        }
    }

    pub fn grow(&mut self, amount: Size, new_state: bool) {
        let unused_trailing_bits = self.blocks.len() as u64 * BLOCK_SIZE - self.len.bytes();
        if amount.bytes() > unused_trailing_bits {
            let additional_blocks = amount.bytes() / BLOCK_SIZE + 1;
            assert_eq!(additional_blocks as usize as u64, additional_blocks);
            self.blocks.extend(
                iter::repeat(0).take(additional_blocks as usize),
            );
        }
        let start = self.len;
        self.len += amount;
        self.set_range_inbounds(start, start + amount, new_state);
    }
}

#[inline]
fn bit_index(bits: Size) -> (usize, usize) {
    let bits = bits.bytes();
    let a = bits / BLOCK_SIZE;
    let b = bits % BLOCK_SIZE;
    assert_eq!(a as usize as u64, a);
    assert_eq!(b as usize as u64, b);
    (a as usize, b as usize)
}
