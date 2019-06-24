//! The virtual memory representation of the MIR interpreter.

use super::{
    Pointer, InterpResult, AllocId, ScalarMaybeUndef, write_target_uint, read_target_uint, Scalar,
};

use crate::ty::layout::{Size, Align};
use syntax::ast::Mutability;
use std::iter;
use crate::mir;
use std::ops::{Range, Deref, DerefMut};
use rustc_data_structures::sorted_map::SortedMap;
use rustc_target::abi::HasDataLayout;
use std::borrow::Cow;

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


pub trait AllocationExtra<Tag>: ::std::fmt::Debug + Clone {
    // There is no constructor in here because the constructor's type depends
    // on `MemoryKind`, and making things sufficiently generic leads to painful
    // inference failure.

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
    ) -> InterpResult<'tcx> {
        Ok(())
    }

    /// Hook for performing extra checks on a memory write access.
    #[inline(always)]
    fn memory_written(
        _alloc: &mut Allocation<Tag, Self>,
        _ptr: Pointer<Tag>,
        _size: Size,
    ) -> InterpResult<'tcx> {
        Ok(())
    }

    /// Hook for performing extra checks on a memory deallocation.
    /// `size` will be the size of the allocation.
    #[inline(always)]
    fn memory_deallocated(
        _alloc: &mut Allocation<Tag, Self>,
        _ptr: Pointer<Tag>,
        _size: Size,
    ) -> InterpResult<'tcx> {
        Ok(())
    }
}

// For Tag=() and no extra state, we have is a trivial implementation.
impl AllocationExtra<()> for () { }

// The constructors are all without extra; the extra gets added by a machine hook later.
impl<Tag> Allocation<Tag> {
    /// Creates a read-only allocation initialized by the given bytes
    pub fn from_bytes<'a>(slice: impl Into<Cow<'a, [u8]>>, align: Align) -> Self {
        let bytes = slice.into().into_owned();
        let undef_mask = UndefMask::new(Size::from_bytes(bytes.len() as u64), true);
        Self {
            bytes,
            relocations: Relocations::new(),
            undef_mask,
            align,
            mutability: Mutability::Immutable,
            extra: (),
        }
    }

    pub fn from_byte_aligned_bytes<'a>(slice: impl Into<Cow<'a, [u8]>>) -> Self {
        Allocation::from_bytes(slice, Align::from_bytes(1).unwrap())
    }

    pub fn undef(size: Size, align: Align) -> Self {
        assert_eq!(size.bytes() as usize as u64, size.bytes());
        Allocation {
            bytes: vec![0; size.bytes() as usize],
            relocations: Relocations::new(),
            undef_mask: UndefMask::new(size, false),
            align,
            mutability: Mutability::Mutable,
            extra: (),
        }
    }
}

impl<'tcx> ::serialize::UseSpecializedDecodable for &'tcx Allocation {}

/// Byte accessors
impl<'tcx, Tag: Copy, Extra: AllocationExtra<Tag>> Allocation<Tag, Extra> {
    /// Just a small local helper function to avoid a bit of code repetition.
    /// Returns the range of this allocation that was meant.
    #[inline]
    fn check_bounds(
        &self,
        offset: Size,
        size: Size
    ) -> Range<usize> {
        let end = offset + size; // this does overflow checking
        assert_eq!(
            end.bytes() as usize as u64, end.bytes(),
            "cannot handle this access on this host architecture"
        );
        let end = end.bytes() as usize;
        assert!(
            end <= self.bytes.len(),
            "Out-of-bounds access at offset {}, size {} in allocation of size {}",
            offset.bytes(), size.bytes(), self.bytes.len()
        );
        (offset.bytes() as usize)..end
    }

    /// The last argument controls whether we error out when there are undefined
    /// or pointer bytes. You should never call this, call `get_bytes` or
    /// `get_bytes_with_undef_and_ptr` instead,
    ///
    /// This function also guarantees that the resulting pointer will remain stable
    /// even when new allocations are pushed to the `HashMap`. `copy_repeatedly` relies
    /// on that.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    fn get_bytes_internal(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
        check_defined_and_ptr: bool,
    ) -> InterpResult<'tcx, &[u8]>
    {
        let range = self.check_bounds(ptr.offset, size);

        if check_defined_and_ptr {
            self.check_defined(ptr, size)?;
            self.check_relocations(cx, ptr, size)?;
        } else {
            // We still don't want relocations on the *edges*
            self.check_relocation_edges(cx, ptr, size)?;
        }

        AllocationExtra::memory_read(self, ptr, size)?;

        Ok(&self.bytes[range])
    }

    /// Check that these bytes are initialized and not pointer bytes, and then return them
    /// as a slice.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    #[inline]
    pub fn get_bytes(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx, &[u8]>
    {
        self.get_bytes_internal(cx, ptr, size, true)
    }

    /// It is the caller's responsibility to handle undefined and pointer bytes.
    /// However, this still checks that there are no relocations on the *edges*.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    #[inline]
    pub fn get_bytes_with_undef_and_ptr(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx, &[u8]>
    {
        self.get_bytes_internal(cx, ptr, size, false)
    }

    /// Just calling this already marks everything as defined and removes relocations,
    /// so be sure to actually put data there!
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    pub fn get_bytes_mut(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx, &mut [u8]>
    {
        let range = self.check_bounds(ptr.offset, size);

        self.mark_definedness(ptr, size, true);
        self.clear_relocations(cx, ptr, size)?;

        AllocationExtra::memory_written(self, ptr, size)?;

        Ok(&mut self.bytes[range])
    }
}

/// Reading and writing
impl<'tcx, Tag: Copy, Extra: AllocationExtra<Tag>> Allocation<Tag, Extra> {
    /// Reads bytes until a `0` is encountered. Will error if the end of the allocation is reached
    /// before a `0` is found.
    pub fn read_c_str(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
    ) -> InterpResult<'tcx, &[u8]>
    {
        assert_eq!(ptr.offset.bytes() as usize as u64, ptr.offset.bytes());
        let offset = ptr.offset.bytes() as usize;
        match self.bytes[offset..].iter().position(|&c| c == 0) {
            Some(size) => {
                let size_with_null = Size::from_bytes((size + 1) as u64);
                // Go through `get_bytes` for checks and AllocationExtra hooks.
                // We read the null, so we include it in the request, but we want it removed
                // from the result, so we do subslicing.
                Ok(&self.get_bytes(cx, ptr, size_with_null)?[..size])
            }
            // This includes the case where `offset` is out-of-bounds to begin with.
            None => err!(UnterminatedCString(ptr.erase_tag())),
        }
    }

    /// Validates that `ptr.offset` and `ptr.offset + size` do not point to the middle of a
    /// relocation. If `allow_ptr_and_undef` is `false`, also enforces that the memory in the
    /// given range contains neither relocations nor undef bytes.
    pub fn check_bytes(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
        allow_ptr_and_undef: bool,
    ) -> InterpResult<'tcx>
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
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    pub fn write_bytes(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        src: &[u8],
    ) -> InterpResult<'tcx>
    {
        let bytes = self.get_bytes_mut(cx, ptr, Size::from_bytes(src.len() as u64))?;
        bytes.clone_from_slice(src);
        Ok(())
    }

    /// Sets `count` bytes starting at `ptr.offset` with `val`. Basically `memset`.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    pub fn write_repeat(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        val: u8,
        count: Size
    ) -> InterpResult<'tcx>
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
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    pub fn read_scalar(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size
    ) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>>
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

    /// Read a pointer-sized scalar.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    pub fn read_ptr_sized(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
    ) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>>
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
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    pub fn write_scalar(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        val: ScalarMaybeUndef<Tag>,
        type_size: Size,
    ) -> InterpResult<'tcx>
    {
        let val = match val {
            ScalarMaybeUndef::Scalar(scalar) => scalar,
            ScalarMaybeUndef::Undef => {
                self.mark_definedness(ptr, type_size, false);
                return Ok(());
            },
        };

        let bytes = match val.to_bits_or_ptr(type_size, cx) {
            Err(val) => val.offset.bytes() as u128,
            Ok(data) => data,
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

    /// Write a pointer-sized scalar.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    pub fn write_ptr_sized(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        val: ScalarMaybeUndef<Tag>
    ) -> InterpResult<'tcx>
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
    ) -> InterpResult<'tcx> {
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
    ) -> InterpResult<'tcx> {
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
    ) -> InterpResult<'tcx> {
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
    fn check_defined(&self, ptr: Pointer<Tag>, size: Size) -> InterpResult<'tcx> {
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
    ) {
        if size.bytes() == 0 {
            return;
        }
        self.undef_mask.set_range(
            ptr.offset,
            ptr.offset + size,
            new_state,
        );
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

/// A bitmask where each bit refers to the byte with the same index. If the bit is `true`, the byte
/// is defined. If it is `false` the byte is undefined.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable)]
pub struct UndefMask {
    blocks: Vec<Block>,
    len: Size,
}

impl_stable_hash_for!(struct mir::interpret::UndefMask{blocks, len});

impl UndefMask {
    pub const BLOCK_SIZE: u64 = 64;

    pub fn new(size: Size, state: bool) -> Self {
        let mut m = UndefMask {
            blocks: vec![],
            len: Size::ZERO,
        };
        m.grow(size, state);
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

        // FIXME(oli-obk): optimize this for allocations larger than a block.
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
        let (blocka, bita) = bit_index(start);
        let (blockb, bitb) = bit_index(end);
        if blocka == blockb {
            // first set all bits but the first `bita`
            // then unset the last `64 - bitb` bits
            let range = if bitb == 0 {
                u64::max_value() << bita
            } else {
                (u64::max_value() << bita) & (u64::max_value() >> (64 - bitb))
            };
            if new_state {
                self.blocks[blocka] |= range;
            } else {
                self.blocks[blocka] &= !range;
            }
            return;
        }
        // across block boundaries
        if new_state {
            // set bita..64 to 1
            self.blocks[blocka] |= u64::max_value() << bita;
            // set 0..bitb to 1
            if bitb != 0 {
                self.blocks[blockb] |= u64::max_value() >> (64 - bitb);
            }
            // fill in all the other blocks (much faster than one bit at a time)
            for block in (blocka + 1) .. blockb {
                self.blocks[block] = u64::max_value();
            }
        } else {
            // set bita..64 to 0
            self.blocks[blocka] &= !(u64::max_value() << bita);
            // set 0..bitb to 0
            if bitb != 0 {
                self.blocks[blockb] &= !(u64::max_value() >> (64 - bitb));
            }
            // fill in all the other blocks (much faster than one bit at a time)
            for block in (blocka + 1) .. blockb {
                self.blocks[block] = 0;
            }
        }
    }

    #[inline]
    pub fn get(&self, i: Size) -> bool {
        let (block, bit) = bit_index(i);
        (self.blocks[block] & (1 << bit)) != 0
    }

    #[inline]
    pub fn set(&mut self, i: Size, new_state: bool) {
        let (block, bit) = bit_index(i);
        self.set_bit(block, bit, new_state);
    }

    #[inline]
    fn set_bit(&mut self, block: usize, bit: usize, new_state: bool) {
        if new_state {
            self.blocks[block] |= 1 << bit;
        } else {
            self.blocks[block] &= !(1 << bit);
        }
    }

    pub fn grow(&mut self, amount: Size, new_state: bool) {
        if amount.bytes() == 0 {
            return;
        }
        let unused_trailing_bits = self.blocks.len() as u64 * Self::BLOCK_SIZE - self.len.bytes();
        if amount.bytes() > unused_trailing_bits {
            let additional_blocks = amount.bytes() / Self::BLOCK_SIZE + 1;
            assert_eq!(additional_blocks as usize as u64, additional_blocks);
            self.blocks.extend(
                // FIXME(oli-obk): optimize this by repeating `new_state as Block`
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
    let a = bits / UndefMask::BLOCK_SIZE;
    let b = bits % UndefMask::BLOCK_SIZE;
    assert_eq!(a as usize as u64, a);
    assert_eq!(b as usize as u64, b);
    (a as usize, b as usize)
}
