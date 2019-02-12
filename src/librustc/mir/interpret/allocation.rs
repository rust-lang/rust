//! The virtual memory representation of the MIR interpreter.

use super::{
    Pointer, EvalResult, AllocId, ScalarMaybeUndef, write_target_uint, read_target_uint, Scalar,
    truncate,
};

use crate::ty::layout::{Size, Align};
use syntax::ast::Mutability;
use std::iter;
use crate::mir;
use std::ops::{Deref, DerefMut};
use rustc_data_structures::sorted_map::SortedMap;
use rustc_target::abi::HasDataLayout;

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


pub trait AllocationExtra<Tag, MemoryExtra>: ::std::fmt::Debug + Clone {
    /// Hook to initialize the extra data when an allocation gets created.
    fn memory_allocated(
        _size: Size,
        _memory_extra: &MemoryExtra
    ) -> Self;

    /// Hook for performing extra checks on a memory read access.
    ///
    /// Takes read-only access to the allocation so we can keep all the memory read
    /// operations take `&self`. Use a `RefCell` in `AllocExtra` if you
    /// need to mutate.
    #[inline(always)]
    fn memory_read(
        _alloc: &Allocation<Tag, Self>,
        _ptr: Pointer<Tag>,
        _size: Size,
    ) -> EvalResult<'tcx> {
        Ok(())
    }

    /// Hook for performing extra checks on a memory write access.
    #[inline(always)]
    fn memory_written(
        _alloc: &mut Allocation<Tag, Self>,
        _ptr: Pointer<Tag>,
        _size: Size,
    ) -> EvalResult<'tcx> {
        Ok(())
    }

    /// Hook for performing extra checks on a memory deallocation.
    /// `size` will be the size of the allocation.
    #[inline(always)]
    fn memory_deallocated(
        _alloc: &mut Allocation<Tag, Self>,
        _ptr: Pointer<Tag>,
        _size: Size,
    ) -> EvalResult<'tcx> {
        Ok(())
    }
}

impl AllocationExtra<(), ()> for () {
    #[inline(always)]
    fn memory_allocated(
        _size: Size,
        _memory_extra: &()
    ) -> Self {
        ()
    }
}

impl<Tag, Extra> Allocation<Tag, Extra> {
    /// Creates a read-only allocation initialized by the given bytes
    pub fn from_bytes(slice: &[u8], align: Align, extra: Extra) -> Self {
        let mut undef_mask = UndefMask::new(Size::ZERO);
        undef_mask.grow(Size::from_bytes(slice.len() as u64), true);
        Self {
            bytes: slice.to_owned(),
            relocations: Relocations::new(),
            undef_mask,
            align,
            mutability: Mutability::Immutable,
            extra,
        }
    }

    pub fn from_byte_aligned_bytes(slice: &[u8], extra: Extra) -> Self {
        Allocation::from_bytes(slice, Align::from_bytes(1).unwrap(), extra)
    }

    pub fn undef(size: Size, align: Align, extra: Extra) -> Self {
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        Allocation {
            bytes: vec![0; size.bytes() as usize],
            relocations: Relocations::new(),
            undef_mask: UndefMask::new(size),
            align,
            mutability: Mutability::Mutable,
            extra,
        }
    }
}

impl<'tcx> ::serialize::UseSpecializedDecodable for &'tcx Allocation {}

/// Alignment and bounds checks
impl<'tcx, Tag, Extra> Allocation<Tag, Extra> {
    /// Checks if the pointer is "in-bounds". Notice that a pointer pointing at the end
    /// of an allocation (i.e., at the first *inaccessible* location) *is* considered
    /// in-bounds!  This follows C's/LLVM's rules.
    /// If you want to check bounds before doing a memory access, better use `check_bounds`.
    fn check_bounds_ptr(
        &self,
        ptr: Pointer<Tag>,
    ) -> EvalResult<'tcx> {
        let allocation_size = self.bytes.len() as u64;
        ptr.check_in_alloc(Size::from_bytes(allocation_size), InboundsCheck::Live)
    }

    /// Checks if the memory range beginning at `ptr` and of size `Size` is "in-bounds".
    #[inline(always)]
    pub fn check_bounds(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> EvalResult<'tcx> {
        // if ptr.offset is in bounds, then so is ptr (because offset checks for overflow)
        self.check_bounds_ptr(ptr.offset(size, cx)?)
    }
}

/// Byte accessors
impl<'tcx, Tag: Copy, Extra> Allocation<Tag, Extra> {
    /// The last argument controls whether we error out when there are undefined
    /// or pointer bytes. You should never call this, call `get_bytes` or
    /// `get_bytes_with_undef_and_ptr` instead,
    ///
    /// This function also guarantees that the resulting pointer will remain stable
    /// even when new allocations are pushed to the `HashMap`. `copy_repeatedly` relies
    /// on that.
    fn get_bytes_internal<MemoryExtra>(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
        check_defined_and_ptr: bool,
    ) -> EvalResult<'tcx, &[u8]>
        // FIXME: Working around https://github.com/rust-lang/rust/issues/56209
        where Extra: AllocationExtra<Tag, MemoryExtra>
    {
        self.check_bounds(cx, ptr, size)?;

        if check_defined_and_ptr {
            self.check_defined(ptr, size)?;
            self.check_relocations(cx, ptr, size)?;
        } else {
            // We still don't want relocations on the *edges*
            self.check_relocation_edges(cx, ptr, size)?;
        }

        AllocationExtra::memory_read(self, ptr, size)?;

        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        let offset = ptr.offset.bytes() as usize;
        Ok(&self.bytes[offset..offset + size.bytes() as usize])
    }

    #[inline]
    pub fn get_bytes<MemoryExtra>(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> EvalResult<'tcx, &[u8]>
        // FIXME: Working around https://github.com/rust-lang/rust/issues/56209
        where Extra: AllocationExtra<Tag, MemoryExtra>
    {
        self.get_bytes_internal(cx, ptr, size, true)
    }

    /// It is the caller's responsibility to handle undefined and pointer bytes.
    /// However, this still checks that there are no relocations on the *edges*.
    #[inline]
    pub fn get_bytes_with_undef_and_ptr<MemoryExtra>(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> EvalResult<'tcx, &[u8]>
        // FIXME: Working around https://github.com/rust-lang/rust/issues/56209
        where Extra: AllocationExtra<Tag, MemoryExtra>
    {
        self.get_bytes_internal(cx, ptr, size, false)
    }

    /// Just calling this already marks everything as defined and removes relocations,
    /// so be sure to actually put data there!
    pub fn get_bytes_mut<MemoryExtra>(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> EvalResult<'tcx, &mut [u8]>
        // FIXME: Working around https://github.com/rust-lang/rust/issues/56209
        where Extra: AllocationExtra<Tag, MemoryExtra>
    {
        assert_ne!(size.bytes(), 0, "0-sized accesses should never even get a `Pointer`");
        self.check_bounds(cx, ptr, size)?;

        self.mark_definedness(ptr, size, true)?;
        self.clear_relocations(cx, ptr, size)?;

        AllocationExtra::memory_written(self, ptr, size)?;

        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        let offset = ptr.offset.bytes() as usize;
        Ok(&mut self.bytes[offset..offset + size.bytes() as usize])
    }
}

/// Reading and writing
impl<'tcx, Tag: Copy, Extra> Allocation<Tag, Extra> {
    /// Reads bytes until a `0` is encountered. Will error if the end of the allocation is reached
    /// before a `0` is found.
    pub fn read_c_str<MemoryExtra>(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
    ) -> EvalResult<'tcx, &[u8]>
        // FIXME: Working around https://github.com/rust-lang/rust/issues/56209
        where Extra: AllocationExtra<Tag, MemoryExtra>
    {
        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        let offset = ptr.offset.bytes() as usize;
        match self.bytes[offset..].iter().position(|&c| c == 0) {
            Some(size) => {
                let size_with_null = Size::from_bytes((size + 1) as u64);
                // Go through `get_bytes` for checks and AllocationExtra hooks.
                // We read the null, so we include it in the request, but we want it removed
                // from the result!
                Ok(&self.get_bytes(cx, ptr, size_with_null)?[..size])
            }
            None => err!(UnterminatedCString(ptr.erase_tag())),
        }
    }

    /// Validates that `ptr.offset` and `ptr.offset + size` do not point to the middle of a
    /// relocation. If `allow_ptr_and_undef` is `false`, also enforces that the memory in the
    /// given range contains neither relocations nor undef bytes.
    pub fn check_bytes<MemoryExtra>(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
        allow_ptr_and_undef: bool,
    ) -> EvalResult<'tcx>
        // FIXME: Working around https://github.com/rust-lang/rust/issues/56209
        where Extra: AllocationExtra<Tag, MemoryExtra>
    {
        // Check bounds and relocations on the edges
        self.get_bytes_with_undef_and_ptr(cx, ptr, size)?;
        // Check undef and ptr
        if !allow_ptr_and_undef {
            self.check_defined(ptr, size)?;
            self.check_relocations(cx, ptr, size)?;
        }
        Ok(())
    }

    /// Writes `src` to the memory starting at `ptr.offset`.
    ///
    /// Will do bounds checks on the allocation.
    pub fn write_bytes<MemoryExtra>(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        src: &[u8],
    ) -> EvalResult<'tcx>
        // FIXME: Working around https://github.com/rust-lang/rust/issues/56209
        where Extra: AllocationExtra<Tag, MemoryExtra>
    {
        let bytes = self.get_bytes_mut(cx, ptr, Size::from_bytes(src.len() as u64))?;
        bytes.clone_from_slice(src);
        Ok(())
    }

    /// Sets `count` bytes starting at `ptr.offset` with `val`. Basically `memset`.
    pub fn write_repeat<MemoryExtra>(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        val: u8,
        count: Size
    ) -> EvalResult<'tcx>
        // FIXME: Working around https://github.com/rust-lang/rust/issues/56209
        where Extra: AllocationExtra<Tag, MemoryExtra>
    {
        let bytes = self.get_bytes_mut(cx, ptr, count)?;
        for b in bytes {
            *b = val;
        }
        Ok(())
    }

    /// Read a *non-ZST* scalar
    ///
    /// zsts can't be read out of two reasons:
    /// * byteorder cannot work with zero element buffers
    /// * in oder to obtain a `Pointer` we need to check for ZSTness anyway due to integer pointers
    ///   being valid for ZSTs
    ///
    /// Note: This function does not do *any* alignment checks, you need to do these before calling
    pub fn read_scalar<MemoryExtra>(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size
    ) -> EvalResult<'tcx, ScalarMaybeUndef<Tag>>
        // FIXME: Working around https://github.com/rust-lang/rust/issues/56209
        where Extra: AllocationExtra<Tag, MemoryExtra>
    {
        // get_bytes_unchecked tests relocation edges
        let bytes = self.get_bytes_with_undef_and_ptr(cx, ptr, size)?;
        // Undef check happens *after* we established that the alignment is correct.
        // We must not return Ok() for unaligned pointers!
        if self.check_defined(ptr, size).is_err() {
            // this inflates undefined bytes to the entire scalar, even if only a few
            // bytes are undefined
            return Ok(ScalarMaybeUndef::Undef);
        }
        // Now we do the actual reading
        let bits = read_target_uint(cx.data_layout().endian, bytes).unwrap();
        // See if we got a pointer
        if size != cx.data_layout().pointer_size {
            // *Now* better make sure that the inside also is free of relocations.
            self.check_relocations(cx, ptr, size)?;
        } else {
            match self.relocations.get(&ptr.offset) {
                Some(&(tag, alloc_id)) => {
                    let ptr = Pointer::new_with_tag(alloc_id, Size::from_bytes(bits as u64), tag);
                    return Ok(ScalarMaybeUndef::Scalar(ptr.into()))
                }
                None => {},
            }
        }
        // We don't. Just return the bits.
        Ok(ScalarMaybeUndef::Scalar(Scalar::from_uint(bits, size)))
    }

    /// Note: This function does not do *any* alignment checks, you need to do these before calling
    pub fn read_ptr_sized<MemoryExtra>(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
    ) -> EvalResult<'tcx, ScalarMaybeUndef<Tag>>
        // FIXME: Working around https://github.com/rust-lang/rust/issues/56209
        where Extra: AllocationExtra<Tag, MemoryExtra>
    {
        self.read_scalar(cx, ptr, cx.data_layout().pointer_size)
    }

    /// Write a *non-ZST* scalar
    ///
    /// zsts can't be read out of two reasons:
    /// * byteorder cannot work with zero element buffers
    /// * in oder to obtain a `Pointer` we need to check for ZSTness anyway due to integer pointers
    ///   being valid for ZSTs
    ///
    /// Note: This function does not do *any* alignment checks, you need to do these before calling
    pub fn write_scalar<MemoryExtra>(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        val: ScalarMaybeUndef<Tag>,
        type_size: Size,
    ) -> EvalResult<'tcx>
        // FIXME: Working around https://github.com/rust-lang/rust/issues/56209
        where Extra: AllocationExtra<Tag, MemoryExtra>
    {
        let val = match val {
            ScalarMaybeUndef::Scalar(scalar) => scalar,
            ScalarMaybeUndef::Undef => return self.mark_definedness(ptr, type_size, false),
        };

        let bytes = match val {
            Scalar::Ptr(val) => {
                assert_eq!(type_size, cx.data_layout().pointer_size);
                val.offset.bytes() as u128
            }

            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, type_size.bytes());
                debug_assert_eq!(truncate(bits, Size::from_bytes(size.into())), bits,
                    "Unexpected value of size {} when writing to memory", size);
                bits
            },
        };

        let endian = cx.data_layout().endian;
        let dst = self.get_bytes_mut(cx, ptr, type_size)?;
        write_target_uint(endian, dst, bytes).unwrap();

        // See if we have to also write a relocation
        match val {
            Scalar::Ptr(val) => {
                self.relocations.insert(
                    ptr.offset,
                    (val.tag, val.alloc_id),
                );
            }
            _ => {}
        }

        Ok(())
    }

    /// Note: This function does not do *any* alignment checks, you need to do these before calling
    pub fn write_ptr_sized<MemoryExtra>(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        val: ScalarMaybeUndef<Tag>
    ) -> EvalResult<'tcx>
        // FIXME: Working around https://github.com/rust-lang/rust/issues/56209
        where Extra: AllocationExtra<Tag, MemoryExtra>
    {
        let ptr_size = cx.data_layout().pointer_size;
        self.write_scalar(cx, ptr.into(), val, ptr_size)
    }
}

/// Relocations
impl<'tcx, Tag: Copy, Extra> Allocation<Tag, Extra> {
    /// Returns all relocations overlapping with the given ptr-offset pair.
    pub fn relocations(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> &[(Size, (Tag, AllocId))] {
        // We have to go back `pointer_size - 1` bytes, as that one would still overlap with
        // the beginning of this range.
        let start = ptr.offset.bytes().saturating_sub(cx.data_layout().pointer_size.bytes() - 1);
        let end = ptr.offset + size; // this does overflow checking
        self.relocations.range(Size::from_bytes(start)..end)
    }

    /// Checks that there are no relocations overlapping with the given range.
    #[inline(always)]
    fn check_relocations(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> EvalResult<'tcx> {
        if self.relocations(cx, ptr, size).is_empty() {
            Ok(())
        } else {
            err!(ReadPointerAsBytes)
        }
    }

    /// Removes all relocations inside the given range.
    /// If there are relocations overlapping with the edges, they
    /// are removed as well *and* the bytes they cover are marked as
    /// uninitialized. This is a somewhat odd "spooky action at a distance",
    /// but it allows strictly more code to run than if we would just error
    /// immediately in that case.
    fn clear_relocations(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> EvalResult<'tcx> {
        // Find the start and end of the given range and its outermost relocations.
        let (first, last) = {
            // Find all relocations overlapping the given range.
            let relocations = self.relocations(cx, ptr, size);
            if relocations.is_empty() {
                return Ok(());
            }

            (relocations.first().unwrap().0,
             relocations.last().unwrap().0 + cx.data_layout().pointer_size)
        };
        let start = ptr.offset;
        let end = start + size;

        // Mark parts of the outermost relocations as undefined if they partially fall outside the
        // given range.
        if first < start {
            self.undef_mask.set_range(first, start, false);
        }
        if last > end {
            self.undef_mask.set_range(end, last, false);
        }

        // Forget all the relocations.
        self.relocations.remove_range(first..last);

        Ok(())
    }

    /// Error if there are relocations overlapping with the edges of the
    /// given memory range.
    #[inline]
    fn check_relocation_edges(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> EvalResult<'tcx> {
        self.check_relocations(cx, ptr, Size::ZERO)?;
        self.check_relocations(cx, ptr.offset(size, cx)?, Size::ZERO)?;
        Ok(())
    }
}


/// Undefined bytes
impl<'tcx, Tag, Extra> Allocation<Tag, Extra> {
    /// Checks that a range of bytes is defined. If not, returns the `ReadUndefBytes`
    /// error which will report the first byte which is undefined.
    #[inline]
    fn check_defined(&self, ptr: Pointer<Tag>, size: Size) -> EvalResult<'tcx> {
        self.undef_mask.is_range_defined(
            ptr.offset,
            ptr.offset + size,
        ).or_else(|idx| err!(ReadUndefBytes(idx)))
    }

    pub fn mark_definedness(
        &mut self,
        ptr: Pointer<Tag>,
        size: Size,
        new_state: bool,
    ) -> EvalResult<'tcx> {
        if size.bytes() == 0 {
            return Ok(());
        }
        self.undef_mask.set_range(
            ptr.offset,
            ptr.offset + size,
            new_state,
        );
        Ok(())
    }
}

/// Relocations
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

    /// Checks whether the range `start..end` (end-exclusive) is entirely defined.
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
