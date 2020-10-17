//! The virtual memory representation of the MIR interpreter.

use std::borrow::Cow;
use std::convert::TryFrom;
use std::iter;
use std::ops::{Deref, DerefMut, Range};

use rustc_ast::Mutability;
use rustc_data_structures::sorted_map::SortedMap;
use rustc_target::abi::{Align, HasDataLayout, Size};

use super::{
    read_target_uint, write_target_uint, AllocId, InterpResult, Pointer, Scalar, ScalarMaybeUninit,
    UninitBytesAccess,
};

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub struct Allocation<Tag = (), Extra = ()> {
    /// The actual bytes of the allocation.
    /// Note that the bytes of a pointer represent the offset of the pointer.
    bytes: Vec<u8>,
    /// Maps from byte addresses to extra data for each pointer.
    /// Only the first byte of a pointer is inserted into the map; i.e.,
    /// every entry in this map applies to `pointer_size` consecutive bytes starting
    /// at the given offset.
    relocations: Relocations<Tag>,
    /// Denotes which part of this allocation is initialized.
    init_mask: InitMask,
    /// The size of the allocation. Currently, must always equal `bytes.len()`.
    pub size: Size,
    /// The alignment of the allocation to detect unaligned reads.
    /// (`Align` guarantees that this is a power of two.)
    pub align: Align,
    /// `true` if the allocation is mutable.
    /// Also used by codegen to determine if a static should be put into mutable memory,
    /// which happens for `static mut` and `static` with interior mutability.
    pub mutability: Mutability,
    /// Extra state for the machine.
    pub extra: Extra,
}

pub trait AllocationExtra<Tag>: std::fmt::Debug + Clone {
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

// For `Tag = ()` and no extra state, we have a trivial implementation.
impl AllocationExtra<()> for () {}

// The constructors are all without extra; the extra gets added by a machine hook later.
impl<Tag> Allocation<Tag> {
    /// Creates a read-only allocation initialized by the given bytes
    pub fn from_bytes<'a>(slice: impl Into<Cow<'a, [u8]>>, align: Align) -> Self {
        let bytes = slice.into().into_owned();
        let size = Size::from_bytes(bytes.len());
        Self {
            bytes,
            relocations: Relocations::new(),
            init_mask: InitMask::new(size, true),
            size,
            align,
            mutability: Mutability::Not,
            extra: (),
        }
    }

    pub fn from_byte_aligned_bytes<'a>(slice: impl Into<Cow<'a, [u8]>>) -> Self {
        Allocation::from_bytes(slice, Align::from_bytes(1).unwrap())
    }

    pub fn uninit(size: Size, align: Align) -> Self {
        Allocation {
            bytes: vec![0; size.bytes_usize()],
            relocations: Relocations::new(),
            init_mask: InitMask::new(size, false),
            size,
            align,
            mutability: Mutability::Mut,
            extra: (),
        }
    }
}

impl Allocation<(), ()> {
    /// Add Tag and Extra fields
    pub fn with_tags_and_extra<T, E>(
        self,
        mut tagger: impl FnMut(AllocId) -> T,
        extra: E,
    ) -> Allocation<T, E> {
        Allocation {
            bytes: self.bytes,
            size: self.size,
            relocations: Relocations::from_presorted(
                self.relocations
                    .iter()
                    // The allocations in the relocations (pointers stored *inside* this allocation)
                    // all get the base pointer tag.
                    .map(|&(offset, ((), alloc))| {
                        let tag = tagger(alloc);
                        (offset, (tag, alloc))
                    })
                    .collect(),
            ),
            init_mask: self.init_mask,
            align: self.align,
            mutability: self.mutability,
            extra,
        }
    }
}

/// Raw accessors. Provide access to otherwise private bytes.
impl<Tag, Extra> Allocation<Tag, Extra> {
    pub fn len(&self) -> usize {
        self.size.bytes_usize()
    }

    /// Looks at a slice which may describe uninitialized bytes or describe a relocation. This differs
    /// from `get_bytes_with_uninit_and_ptr` in that it does no relocation checks (even on the
    /// edges) at all. It further ignores `AllocationExtra` callbacks.
    /// This must not be used for reads affecting the interpreter execution.
    pub fn inspect_with_uninit_and_ptr_outside_interpreter(&self, range: Range<usize>) -> &[u8] {
        &self.bytes[range]
    }

    /// Returns the mask indicating which bytes are initialized.
    pub fn init_mask(&self) -> &InitMask {
        &self.init_mask
    }

    /// Returns the relocation list.
    pub fn relocations(&self) -> &Relocations<Tag> {
        &self.relocations
    }
}

/// Byte accessors.
impl<'tcx, Tag: Copy, Extra: AllocationExtra<Tag>> Allocation<Tag, Extra> {
    /// Just a small local helper function to avoid a bit of code repetition.
    /// Returns the range of this allocation that was meant.
    #[inline]
    fn check_bounds(&self, offset: Size, size: Size) -> Range<usize> {
        let end = offset + size; // This does overflow checking.
        let end = usize::try_from(end.bytes()).expect("access too big for this host architecture");
        assert!(
            end <= self.len(),
            "Out-of-bounds access at offset {}, size {} in allocation of size {}",
            offset.bytes(),
            size.bytes(),
            self.len()
        );
        offset.bytes_usize()..end
    }

    /// The last argument controls whether we error out when there are uninitialized
    /// or pointer bytes. You should never call this, call `get_bytes` or
    /// `get_bytes_with_uninit_and_ptr` instead,
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
        check_init_and_ptr: bool,
    ) -> InterpResult<'tcx, &[u8]> {
        let range = self.check_bounds(ptr.offset, size);

        if check_init_and_ptr {
            self.check_init(ptr, size)?;
            self.check_relocations(cx, ptr, size)?;
        } else {
            // We still don't want relocations on the *edges*.
            self.check_relocation_edges(cx, ptr, size)?;
        }

        AllocationExtra::memory_read(self, ptr, size)?;

        Ok(&self.bytes[range])
    }

    /// Checks that these bytes are initialized and not pointer bytes, and then return them
    /// as a slice.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    /// Most likely, you want to use the `PlaceTy` and `OperandTy`-based methods
    /// on `InterpCx` instead.
    #[inline]
    pub fn get_bytes(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx, &[u8]> {
        self.get_bytes_internal(cx, ptr, size, true)
    }

    /// It is the caller's responsibility to handle uninitialized and pointer bytes.
    /// However, this still checks that there are no relocations on the *edges*.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    #[inline]
    pub fn get_bytes_with_uninit_and_ptr(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx, &[u8]> {
        self.get_bytes_internal(cx, ptr, size, false)
    }

    /// Just calling this already marks everything as defined and removes relocations,
    /// so be sure to actually put data there!
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    /// Most likely, you want to use the `PlaceTy` and `OperandTy`-based methods
    /// on `InterpCx` instead.
    pub fn get_bytes_mut(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx, &mut [u8]> {
        let range = self.check_bounds(ptr.offset, size);

        self.mark_init(ptr, size, true);
        self.clear_relocations(cx, ptr, size)?;

        AllocationExtra::memory_written(self, ptr, size)?;

        Ok(&mut self.bytes[range])
    }
}

/// Reading and writing.
impl<'tcx, Tag: Copy, Extra: AllocationExtra<Tag>> Allocation<Tag, Extra> {
    /// Reads bytes until a `0` is encountered. Will error if the end of the allocation is reached
    /// before a `0` is found.
    ///
    /// Most likely, you want to call `Memory::read_c_str` instead of this method.
    pub fn read_c_str(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
    ) -> InterpResult<'tcx, &[u8]> {
        let offset = ptr.offset.bytes_usize();
        Ok(match self.bytes[offset..].iter().position(|&c| c == 0) {
            Some(size) => {
                let size_with_null = Size::from_bytes(size) + Size::from_bytes(1);
                // Go through `get_bytes` for checks and AllocationExtra hooks.
                // We read the null, so we include it in the request, but we want it removed
                // from the result, so we do subslicing.
                &self.get_bytes(cx, ptr, size_with_null)?[..size]
            }
            // This includes the case where `offset` is out-of-bounds to begin with.
            None => throw_ub!(UnterminatedCString(ptr.erase_tag())),
        })
    }

    /// Validates that `ptr.offset` and `ptr.offset + size` do not point to the middle of a
    /// relocation. If `allow_uninit_and_ptr` is `false`, also enforces that the memory in the
    /// given range contains neither relocations nor uninitialized bytes.
    pub fn check_bytes(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
        allow_uninit_and_ptr: bool,
    ) -> InterpResult<'tcx> {
        // Check bounds and relocations on the edges.
        self.get_bytes_with_uninit_and_ptr(cx, ptr, size)?;
        // Check uninit and ptr.
        if !allow_uninit_and_ptr {
            self.check_init(ptr, size)?;
            self.check_relocations(cx, ptr, size)?;
        }
        Ok(())
    }

    /// Writes `src` to the memory starting at `ptr.offset`.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    /// Most likely, you want to call `Memory::write_bytes` instead of this method.
    pub fn write_bytes(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        src: impl IntoIterator<Item = u8>,
    ) -> InterpResult<'tcx> {
        let mut src = src.into_iter();
        let (lower, upper) = src.size_hint();
        let len = upper.expect("can only write bounded iterators");
        assert_eq!(lower, len, "can only write iterators with a precise length");
        let bytes = self.get_bytes_mut(cx, ptr, Size::from_bytes(len))?;
        // `zip` would stop when the first iterator ends; we want to definitely
        // cover all of `bytes`.
        for dest in bytes {
            *dest = src.next().expect("iterator was shorter than it said it would be");
        }
        src.next().expect_none("iterator was longer than it said it would be");
        Ok(())
    }

    /// Reads a *non-ZST* scalar.
    ///
    /// ZSTs can't be read because in order to obtain a `Pointer`, we need to check
    /// for ZSTness anyway due to integer pointers being valid for ZSTs.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    /// Most likely, you want to call `InterpCx::read_scalar` instead of this method.
    pub fn read_scalar(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> InterpResult<'tcx, ScalarMaybeUninit<Tag>> {
        // `get_bytes_unchecked` tests relocation edges.
        let bytes = self.get_bytes_with_uninit_and_ptr(cx, ptr, size)?;
        // Uninit check happens *after* we established that the alignment is correct.
        // We must not return `Ok()` for unaligned pointers!
        if self.is_init(ptr, size).is_err() {
            // This inflates uninitialized bytes to the entire scalar, even if only a few
            // bytes are uninitialized.
            return Ok(ScalarMaybeUninit::Uninit);
        }
        // Now we do the actual reading.
        let bits = read_target_uint(cx.data_layout().endian, bytes).unwrap();
        // See if we got a pointer.
        if size != cx.data_layout().pointer_size {
            // *Now*, we better make sure that the inside is free of relocations too.
            self.check_relocations(cx, ptr, size)?;
        } else {
            if let Some(&(tag, alloc_id)) = self.relocations.get(&ptr.offset) {
                let ptr = Pointer::new_with_tag(alloc_id, Size::from_bytes(bits), tag);
                return Ok(ScalarMaybeUninit::Scalar(ptr.into()));
            }
        }
        // We don't. Just return the bits.
        Ok(ScalarMaybeUninit::Scalar(Scalar::from_uint(bits, size)))
    }

    /// Reads a pointer-sized scalar.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    /// Most likely, you want to call `InterpCx::read_scalar` instead of this method.
    pub fn read_ptr_sized(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
    ) -> InterpResult<'tcx, ScalarMaybeUninit<Tag>> {
        self.read_scalar(cx, ptr, cx.data_layout().pointer_size)
    }

    /// Writes a *non-ZST* scalar.
    ///
    /// ZSTs can't be read because in order to obtain a `Pointer`, we need to check
    /// for ZSTness anyway due to integer pointers being valid for ZSTs.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    /// Most likely, you want to call `InterpCx::write_scalar` instead of this method.
    pub fn write_scalar(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        val: ScalarMaybeUninit<Tag>,
        type_size: Size,
    ) -> InterpResult<'tcx> {
        let val = match val {
            ScalarMaybeUninit::Scalar(scalar) => scalar,
            ScalarMaybeUninit::Uninit => {
                self.mark_init(ptr, type_size, false);
                return Ok(());
            }
        };

        let bytes = match val.to_bits_or_ptr(type_size, cx) {
            Err(val) => u128::from(val.offset.bytes()),
            Ok(data) => data,
        };

        let endian = cx.data_layout().endian;
        let dst = self.get_bytes_mut(cx, ptr, type_size)?;
        write_target_uint(endian, dst, bytes).unwrap();

        // See if we have to also write a relocation.
        if let Scalar::Ptr(val) = val {
            self.relocations.insert(ptr.offset, (val.tag, val.alloc_id));
        }

        Ok(())
    }

    /// Writes a pointer-sized scalar.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    /// Most likely, you want to call `InterpCx::write_scalar` instead of this method.
    pub fn write_ptr_sized(
        &mut self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        val: ScalarMaybeUninit<Tag>,
    ) -> InterpResult<'tcx> {
        let ptr_size = cx.data_layout().pointer_size;
        self.write_scalar(cx, ptr, val, ptr_size)
    }
}

/// Relocations.
impl<'tcx, Tag: Copy, Extra> Allocation<Tag, Extra> {
    /// Returns all relocations overlapping with the given pointer-offset pair.
    pub fn get_relocations(
        &self,
        cx: &impl HasDataLayout,
        ptr: Pointer<Tag>,
        size: Size,
    ) -> &[(Size, (Tag, AllocId))] {
        // We have to go back `pointer_size - 1` bytes, as that one would still overlap with
        // the beginning of this range.
        let start = ptr.offset.bytes().saturating_sub(cx.data_layout().pointer_size.bytes() - 1);
        let end = ptr.offset + size; // This does overflow checking.
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
        if self.get_relocations(cx, ptr, size).is_empty() {
            Ok(())
        } else {
            throw_unsup!(ReadPointerAsBytes)
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
            let relocations = self.get_relocations(cx, ptr, size);
            if relocations.is_empty() {
                return Ok(());
            }

            (
                relocations.first().unwrap().0,
                relocations.last().unwrap().0 + cx.data_layout().pointer_size,
            )
        };
        let start = ptr.offset;
        let end = start + size; // `Size` addition

        // Mark parts of the outermost relocations as uninitialized if they partially fall outside the
        // given range.
        if first < start {
            self.init_mask.set_range(first, start, false);
        }
        if last > end {
            self.init_mask.set_range(end, last, false);
        }

        // Forget all the relocations.
        self.relocations.remove_range(first..last);

        Ok(())
    }

    /// Errors if there are relocations overlapping with the edges of the
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

/// Uninitialized bytes.
impl<'tcx, Tag: Copy, Extra> Allocation<Tag, Extra> {
    /// Checks whether the given range  is entirely initialized.
    ///
    /// Returns `Ok(())` if it's initialized. Otherwise returns the range of byte
    /// indexes of the first contiguous uninitialized access.
    fn is_init(&self, ptr: Pointer<Tag>, size: Size) -> Result<(), Range<Size>> {
        self.init_mask.is_range_initialized(ptr.offset, ptr.offset + size) // `Size` addition
    }

    /// Checks that a range of bytes is initialized. If not, returns the `InvalidUninitBytes`
    /// error which will report the first range of bytes which is uninitialized.
    fn check_init(&self, ptr: Pointer<Tag>, size: Size) -> InterpResult<'tcx> {
        self.is_init(ptr, size).or_else(|idx_range| {
            throw_ub!(InvalidUninitBytes(Some(Box::new(UninitBytesAccess {
                access_ptr: ptr.erase_tag(),
                access_size: size,
                uninit_ptr: Pointer::new(ptr.alloc_id, idx_range.start),
                uninit_size: idx_range.end - idx_range.start, // `Size` subtraction
            }))))
        })
    }

    pub fn mark_init(&mut self, ptr: Pointer<Tag>, size: Size, is_init: bool) {
        if size.bytes() == 0 {
            return;
        }
        self.init_mask.set_range(ptr.offset, ptr.offset + size, is_init);
    }
}

/// Run-length encoding of the uninit mask.
/// Used to copy parts of a mask multiple times to another allocation.
pub struct InitMaskCompressed {
    /// Whether the first range is initialized.
    initial: bool,
    /// The lengths of ranges that are run-length encoded.
    /// The initialization state of the ranges alternate starting with `initial`.
    ranges: smallvec::SmallVec<[u64; 1]>,
}

impl InitMaskCompressed {
    pub fn no_bytes_init(&self) -> bool {
        // The `ranges` are run-length encoded and of alternating initialization state.
        // So if `ranges.len() > 1` then the second block is an initialized range.
        !self.initial && self.ranges.len() == 1
    }
}

/// Transferring the initialization mask to other allocations.
impl<Tag, Extra> Allocation<Tag, Extra> {
    /// Creates a run-length encoding of the initialization mask.
    pub fn compress_uninit_range(&self, src: Pointer<Tag>, size: Size) -> InitMaskCompressed {
        // Since we are copying `size` bytes from `src` to `dest + i * size` (`for i in 0..repeat`),
        // a naive initialization mask copying algorithm would repeatedly have to read the initialization mask from
        // the source and write it to the destination. Even if we optimized the memory accesses,
        // we'd be doing all of this `repeat` times.
        // Therefore we precompute a compressed version of the initialization mask of the source value and
        // then write it back `repeat` times without computing any more information from the source.

        // A precomputed cache for ranges of initialized / uninitialized bits
        // 0000010010001110 will become
        // `[5, 1, 2, 1, 3, 3, 1]`,
        // where each element toggles the state.

        let mut ranges = smallvec::SmallVec::<[u64; 1]>::new();
        let initial = self.init_mask.get(src.offset);
        let mut cur_len = 1;
        let mut cur = initial;

        for i in 1..size.bytes() {
            // FIXME: optimize to bitshift the current uninitialized block's bits and read the top bit.
            if self.init_mask.get(src.offset + Size::from_bytes(i)) == cur {
                cur_len += 1;
            } else {
                ranges.push(cur_len);
                cur_len = 1;
                cur = !cur;
            }
        }

        ranges.push(cur_len);

        InitMaskCompressed { ranges, initial }
    }

    /// Applies multiple instances of the run-length encoding to the initialization mask.
    pub fn mark_compressed_init_range(
        &mut self,
        defined: &InitMaskCompressed,
        dest: Pointer<Tag>,
        size: Size,
        repeat: u64,
    ) {
        // An optimization where we can just overwrite an entire range of initialization
        // bits if they are going to be uniformly `1` or `0`.
        if defined.ranges.len() <= 1 {
            self.init_mask.set_range_inbounds(
                dest.offset,
                dest.offset + size * repeat, // `Size` operations
                defined.initial,
            );
            return;
        }

        for mut j in 0..repeat {
            j *= size.bytes();
            j += dest.offset.bytes();
            let mut cur = defined.initial;
            for range in &defined.ranges {
                let old_j = j;
                j += range;
                self.init_mask.set_range_inbounds(
                    Size::from_bytes(old_j),
                    Size::from_bytes(j),
                    cur,
                );
                cur = !cur;
            }
        }
    }
}

/// Relocations.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, TyEncodable, TyDecodable)]
pub struct Relocations<Tag = (), Id = AllocId>(SortedMap<Size, (Tag, Id)>);

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

/// A partial, owned list of relocations to transfer into another allocation.
pub struct AllocationRelocations<Tag> {
    relative_relocations: Vec<(Size, (Tag, AllocId))>,
}

impl<Tag: Copy, Extra> Allocation<Tag, Extra> {
    pub fn prepare_relocation_copy(
        &self,
        cx: &impl HasDataLayout,
        src: Pointer<Tag>,
        size: Size,
        dest: Pointer<Tag>,
        length: u64,
    ) -> AllocationRelocations<Tag> {
        let relocations = self.get_relocations(cx, src, size);
        if relocations.is_empty() {
            return AllocationRelocations { relative_relocations: Vec::new() };
        }

        let mut new_relocations = Vec::with_capacity(relocations.len() * (length as usize));

        for i in 0..length {
            new_relocations.extend(relocations.iter().map(|&(offset, reloc)| {
                // compute offset for current repetition
                let dest_offset = dest.offset + size * i; // `Size` operations
                (
                    // shift offsets from source allocation to destination allocation
                    (offset + dest_offset) - src.offset, // `Size` operations
                    reloc,
                )
            }));
        }

        AllocationRelocations { relative_relocations: new_relocations }
    }

    /// Applies a relocation copy.
    /// The affected range, as defined in the parameters to `prepare_relocation_copy` is expected
    /// to be clear of relocations.
    pub fn mark_relocation_range(&mut self, relocations: AllocationRelocations<Tag>) {
        self.relocations.insert_presorted(relocations.relative_relocations);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Uninitialized byte tracking
////////////////////////////////////////////////////////////////////////////////

type Block = u64;

/// A bitmask where each bit refers to the byte with the same index. If the bit is `true`, the byte
/// is initialized. If it is `false` the byte is uninitialized.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub struct InitMask {
    blocks: Vec<Block>,
    len: Size,
}

impl InitMask {
    pub const BLOCK_SIZE: u64 = 64;

    pub fn new(size: Size, state: bool) -> Self {
        let mut m = InitMask { blocks: vec![], len: Size::ZERO };
        m.grow(size, state);
        m
    }

    /// Checks whether the range `start..end` (end-exclusive) is entirely initialized.
    ///
    /// Returns `Ok(())` if it's initialized. Otherwise returns a range of byte
    /// indexes for the first contiguous span of the uninitialized access.
    #[inline]
    pub fn is_range_initialized(&self, start: Size, end: Size) -> Result<(), Range<Size>> {
        if end > self.len {
            return Err(self.len..end);
        }

        // FIXME(oli-obk): optimize this for allocations larger than a block.
        let idx = (start.bytes()..end.bytes()).map(Size::from_bytes).find(|&i| !self.get(i));

        match idx {
            Some(idx) => {
                let uninit_end = (idx.bytes()..end.bytes())
                    .map(Size::from_bytes)
                    .find(|&i| self.get(i))
                    .unwrap_or(end);
                Err(idx..uninit_end)
            }
            None => Ok(()),
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
            // First set all bits except the first `bita`,
            // then unset the last `64 - bitb` bits.
            let range = if bitb == 0 {
                u64::MAX << bita
            } else {
                (u64::MAX << bita) & (u64::MAX >> (64 - bitb))
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
            // Set `bita..64` to `1`.
            self.blocks[blocka] |= u64::MAX << bita;
            // Set `0..bitb` to `1`.
            if bitb != 0 {
                self.blocks[blockb] |= u64::MAX >> (64 - bitb);
            }
            // Fill in all the other blocks (much faster than one bit at a time).
            for block in (blocka + 1)..blockb {
                self.blocks[block] = u64::MAX;
            }
        } else {
            // Set `bita..64` to `0`.
            self.blocks[blocka] &= !(u64::MAX << bita);
            // Set `0..bitb` to `0`.
            if bitb != 0 {
                self.blocks[blockb] &= !(u64::MAX >> (64 - bitb));
            }
            // Fill in all the other blocks (much faster than one bit at a time).
            for block in (blocka + 1)..blockb {
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
        let unused_trailing_bits =
            u64::try_from(self.blocks.len()).unwrap() * Self::BLOCK_SIZE - self.len.bytes();
        if amount.bytes() > unused_trailing_bits {
            let additional_blocks = amount.bytes() / Self::BLOCK_SIZE + 1;
            self.blocks.extend(
                // FIXME(oli-obk): optimize this by repeating `new_state as Block`.
                iter::repeat(0).take(usize::try_from(additional_blocks).unwrap()),
            );
        }
        let start = self.len;
        self.len += amount;
        self.set_range_inbounds(start, start + amount, new_state); // `Size` operation
    }
}

#[inline]
fn bit_index(bits: Size) -> (usize, usize) {
    let bits = bits.bytes();
    let a = bits / InitMask::BLOCK_SIZE;
    let b = bits % InitMask::BLOCK_SIZE;
    (usize::try_from(a).unwrap(), usize::try_from(b).unwrap())
}
