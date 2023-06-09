//! The virtual memory representation of the MIR interpreter.

mod init_mask;
mod provenance_map;

use std::borrow::Cow;
use std::fmt;
use std::hash;
use std::hash::Hash;
use std::ops::{Deref, DerefMut, Range};
use std::ptr;

use either::{Left, Right};

use rustc_ast::Mutability;
use rustc_data_structures::intern::Interned;
use rustc_span::DUMMY_SP;
use rustc_target::abi::{Align, HasDataLayout, Size};

use super::{
    read_target_uint, write_target_uint, AllocId, InterpError, InterpResult, Pointer, Provenance,
    ResourceExhaustionInfo, Scalar, ScalarSizeMismatch, UndefinedBehaviorInfo, UninitBytesAccess,
    UnsupportedOpInfo,
};
use crate::ty;
use init_mask::*;
use provenance_map::*;

pub use init_mask::{InitChunk, InitChunkIter};

/// Functionality required for the bytes of an `Allocation`.
pub trait AllocBytes:
    Clone + fmt::Debug + Eq + PartialEq + Hash + Deref<Target = [u8]> + DerefMut<Target = [u8]>
{
    /// Adjust the bytes to the specified alignment -- by default, this is a no-op.
    fn adjust_to_align(self, _align: Align) -> Self;

    /// Create an `AllocBytes` from a slice of `u8`.
    fn from_bytes<'a>(slice: impl Into<Cow<'a, [u8]>>, _align: Align) -> Self;

    /// Create a zeroed `AllocBytes` of the specified size and alignment;
    /// call the callback error handler if there is an error in allocating the memory.
    fn zeroed(size: Size, _align: Align) -> Option<Self>;
}

// Default `bytes` for `Allocation` is a `Box<[u8]>`.
impl AllocBytes for Box<[u8]> {
    fn adjust_to_align(self, _align: Align) -> Self {
        self
    }

    fn from_bytes<'a>(slice: impl Into<Cow<'a, [u8]>>, _align: Align) -> Self {
        Box::<[u8]>::from(slice.into())
    }

    fn zeroed(size: Size, _align: Align) -> Option<Self> {
        let bytes = Box::<[u8]>::try_new_zeroed_slice(size.bytes_usize()).ok()?;
        // SAFETY: the box was zero-allocated, which is a valid initial value for Box<[u8]>
        let bytes = unsafe { bytes.assume_init() };
        Some(bytes)
    }
}

/// This type represents an Allocation in the Miri/CTFE core engine.
///
/// Its public API is rather low-level, working directly with allocation offsets and a custom error
/// type to account for the lack of an AllocId on this level. The Miri/CTFE core engine `memory`
/// module provides higher-level access.
// Note: for performance reasons when interning, some of the `Allocation` fields can be partially
// hashed. (see the `Hash` impl below for more details), so the impl is not derived.
#[derive(Clone, Eq, PartialEq, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub struct Allocation<Prov: Provenance = AllocId, Extra = (), Bytes = Box<[u8]>> {
    /// The actual bytes of the allocation.
    /// Note that the bytes of a pointer represent the offset of the pointer.
    bytes: Bytes,
    /// Maps from byte addresses to extra provenance data for each pointer.
    /// Only the first byte of a pointer is inserted into the map; i.e.,
    /// every entry in this map applies to `pointer_size` consecutive bytes starting
    /// at the given offset.
    provenance: ProvenanceMap<Prov>,
    /// Denotes which part of this allocation is initialized.
    init_mask: InitMask,
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

/// This is the maximum size we will hash at a time, when interning an `Allocation` and its
/// `InitMask`. Note, we hash that amount of bytes twice: at the start, and at the end of a buffer.
/// Used when these two structures are large: we only partially hash the larger fields in that
/// situation. See the comment at the top of their respective `Hash` impl for more details.
const MAX_BYTES_TO_HASH: usize = 64;

/// This is the maximum size (in bytes) for which a buffer will be fully hashed, when interning.
/// Otherwise, it will be partially hashed in 2 slices, requiring at least 2 `MAX_BYTES_TO_HASH`
/// bytes.
const MAX_HASHED_BUFFER_LEN: usize = 2 * MAX_BYTES_TO_HASH;

// Const allocations are only hashed for interning. However, they can be large, making the hashing
// expensive especially since it uses `FxHash`: it's better suited to short keys, not potentially
// big buffers like the actual bytes of allocation. We can partially hash some fields when they're
// large.
impl hash::Hash for Allocation {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        let Self {
            bytes,
            provenance,
            init_mask,
            align,
            mutability,
            extra: (), // don't bother hashing ()
        } = self;

        // Partially hash the `bytes` buffer when it is large. To limit collisions with common
        // prefixes and suffixes, we hash the length and some slices of the buffer.
        let byte_count = bytes.len();
        if byte_count > MAX_HASHED_BUFFER_LEN {
            // Hash the buffer's length.
            byte_count.hash(state);

            // And its head and tail.
            bytes[..MAX_BYTES_TO_HASH].hash(state);
            bytes[byte_count - MAX_BYTES_TO_HASH..].hash(state);
        } else {
            bytes.hash(state);
        }

        // Hash the other fields as usual.
        provenance.hash(state);
        init_mask.hash(state);
        align.hash(state);
        mutability.hash(state);
    }
}

/// Interned types generally have an `Outer` type and an `Inner` type, where
/// `Outer` is a newtype around `Interned<Inner>`, and all the operations are
/// done on `Outer`, because all occurrences are interned. E.g. `Ty` is an
/// outer type and `TyKind` is its inner type.
///
/// Here things are different because only const allocations are interned. This
/// means that both the inner type (`Allocation`) and the outer type
/// (`ConstAllocation`) are used quite a bit.
#[derive(Copy, Clone, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct ConstAllocation<'tcx>(pub Interned<'tcx, Allocation>);

impl<'tcx> fmt::Debug for ConstAllocation<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The debug representation of this is very verbose and basically useless,
        // so don't print it.
        write!(f, "ConstAllocation {{ .. }}")
    }
}

impl<'tcx> ConstAllocation<'tcx> {
    pub fn inner(self) -> &'tcx Allocation {
        self.0.0
    }
}

/// We have our own error type that does not know about the `AllocId`; that information
/// is added when converting to `InterpError`.
#[derive(Debug)]
pub enum AllocError {
    /// A scalar had the wrong size.
    ScalarSizeMismatch(ScalarSizeMismatch),
    /// Encountered a pointer where we needed raw bytes.
    ReadPointerAsBytes,
    /// Partially overwriting a pointer.
    PartialPointerOverwrite(Size),
    /// Partially copying a pointer.
    PartialPointerCopy(Size),
    /// Using uninitialized data where it is not allowed.
    InvalidUninitBytes(Option<UninitBytesAccess>),
}
pub type AllocResult<T = ()> = Result<T, AllocError>;

impl From<ScalarSizeMismatch> for AllocError {
    fn from(s: ScalarSizeMismatch) -> Self {
        AllocError::ScalarSizeMismatch(s)
    }
}

impl AllocError {
    pub fn to_interp_error<'tcx>(self, alloc_id: AllocId) -> InterpError<'tcx> {
        use AllocError::*;
        match self {
            ScalarSizeMismatch(s) => {
                InterpError::UndefinedBehavior(UndefinedBehaviorInfo::ScalarSizeMismatch(s))
            }
            ReadPointerAsBytes => InterpError::Unsupported(UnsupportedOpInfo::ReadPointerAsBytes),
            PartialPointerOverwrite(offset) => InterpError::Unsupported(
                UnsupportedOpInfo::PartialPointerOverwrite(Pointer::new(alloc_id, offset)),
            ),
            PartialPointerCopy(offset) => InterpError::Unsupported(
                UnsupportedOpInfo::PartialPointerCopy(Pointer::new(alloc_id, offset)),
            ),
            InvalidUninitBytes(info) => InterpError::UndefinedBehavior(
                UndefinedBehaviorInfo::InvalidUninitBytes(info.map(|b| (alloc_id, b))),
            ),
        }
    }
}

/// The information that makes up a memory access: offset and size.
#[derive(Copy, Clone)]
pub struct AllocRange {
    pub start: Size,
    pub size: Size,
}

impl fmt::Debug for AllocRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:#x}..{:#x}]", self.start.bytes(), self.end().bytes())
    }
}

/// Free-starting constructor for less syntactic overhead.
#[inline(always)]
pub fn alloc_range(start: Size, size: Size) -> AllocRange {
    AllocRange { start, size }
}

impl From<Range<Size>> for AllocRange {
    #[inline]
    fn from(r: Range<Size>) -> Self {
        alloc_range(r.start, r.end - r.start) // `Size` subtraction (overflow-checked)
    }
}

impl From<Range<usize>> for AllocRange {
    #[inline]
    fn from(r: Range<usize>) -> Self {
        AllocRange::from(Size::from_bytes(r.start)..Size::from_bytes(r.end))
    }
}

impl AllocRange {
    #[inline(always)]
    pub fn end(self) -> Size {
        self.start + self.size // This does overflow checking.
    }

    /// Returns the `subrange` within this range; panics if it is not a subrange.
    #[inline]
    pub fn subrange(self, subrange: AllocRange) -> AllocRange {
        let sub_start = self.start + subrange.start;
        let range = alloc_range(sub_start, subrange.size);
        assert!(range.end() <= self.end(), "access outside the bounds for given AllocRange");
        range
    }
}

// The constructors are all without extra; the extra gets added by a machine hook later.
impl<Prov: Provenance, Bytes: AllocBytes> Allocation<Prov, (), Bytes> {
    /// Creates an allocation from an existing `Bytes` value - this is needed for miri FFI support
    pub fn from_raw_bytes(bytes: Bytes, align: Align, mutability: Mutability) -> Self {
        let size = Size::from_bytes(bytes.len());
        Self {
            bytes,
            provenance: ProvenanceMap::new(),
            init_mask: InitMask::new(size, true),
            align,
            mutability,
            extra: (),
        }
    }

    /// Creates an allocation initialized by the given bytes
    pub fn from_bytes<'a>(
        slice: impl Into<Cow<'a, [u8]>>,
        align: Align,
        mutability: Mutability,
    ) -> Self {
        let bytes = Bytes::from_bytes(slice, align);
        let size = Size::from_bytes(bytes.len());
        Self {
            bytes,
            provenance: ProvenanceMap::new(),
            init_mask: InitMask::new(size, true),
            align,
            mutability,
            extra: (),
        }
    }

    pub fn from_bytes_byte_aligned_immutable<'a>(slice: impl Into<Cow<'a, [u8]>>) -> Self {
        Allocation::from_bytes(slice, Align::ONE, Mutability::Not)
    }

    fn uninit_inner<R>(size: Size, align: Align, fail: impl FnOnce() -> R) -> Result<Self, R> {
        // This results in an error that can happen non-deterministically, since the memory
        // available to the compiler can change between runs. Normally queries are always
        // deterministic. However, we can be non-deterministic here because all uses of const
        // evaluation (including ConstProp!) will make compilation fail (via hard error
        // or ICE) upon encountering a `MemoryExhausted` error.
        let bytes = Bytes::zeroed(size, align).ok_or_else(fail)?;

        Ok(Allocation {
            bytes,
            provenance: ProvenanceMap::new(),
            init_mask: InitMask::new(size, false),
            align,
            mutability: Mutability::Mut,
            extra: (),
        })
    }

    /// Try to create an Allocation of `size` bytes, failing if there is not enough memory
    /// available to the compiler to do so.
    pub fn try_uninit<'tcx>(size: Size, align: Align) -> InterpResult<'tcx, Self> {
        Self::uninit_inner(size, align, || {
            ty::tls::with(|tcx| {
                tcx.sess.delay_span_bug(DUMMY_SP, "exhausted memory during interpretation")
            });
            InterpError::ResourceExhaustion(ResourceExhaustionInfo::MemoryExhausted).into()
        })
    }

    /// Try to create an Allocation of `size` bytes, panics if there is not enough memory
    /// available to the compiler to do so.
    pub fn uninit(size: Size, align: Align) -> Self {
        match Self::uninit_inner(size, align, || {
            panic!("Allocation::uninit called with panic_on_fail had allocation failure");
        }) {
            Ok(x) => x,
            Err(x) => x,
        }
    }
}

impl<Bytes: AllocBytes> Allocation<AllocId, (), Bytes> {
    /// Adjust allocation from the ones in tcx to a custom Machine instance
    /// with a different Provenance and Extra type.
    pub fn adjust_from_tcx<Prov: Provenance, Extra, Err>(
        self,
        cx: &impl HasDataLayout,
        extra: Extra,
        mut adjust_ptr: impl FnMut(Pointer<AllocId>) -> Result<Pointer<Prov>, Err>,
    ) -> Result<Allocation<Prov, Extra, Bytes>, Err> {
        // Compute new pointer provenance, which also adjusts the bytes, and realign the pointer if
        // necessary.
        let mut bytes = self.bytes.adjust_to_align(self.align);

        let mut new_provenance = Vec::with_capacity(self.provenance.ptrs().len());
        let ptr_size = cx.data_layout().pointer_size.bytes_usize();
        let endian = cx.data_layout().endian;
        for &(offset, alloc_id) in self.provenance.ptrs().iter() {
            let idx = offset.bytes_usize();
            let ptr_bytes = &mut bytes[idx..idx + ptr_size];
            let bits = read_target_uint(endian, ptr_bytes).unwrap();
            let (ptr_prov, ptr_offset) =
                adjust_ptr(Pointer::new(alloc_id, Size::from_bytes(bits)))?.into_parts();
            write_target_uint(endian, ptr_bytes, ptr_offset.bytes().into()).unwrap();
            new_provenance.push((offset, ptr_prov));
        }
        // Create allocation.
        Ok(Allocation {
            bytes,
            provenance: ProvenanceMap::from_presorted_ptrs(new_provenance),
            init_mask: self.init_mask,
            align: self.align,
            mutability: self.mutability,
            extra,
        })
    }
}

/// Raw accessors. Provide access to otherwise private bytes.
impl<Prov: Provenance, Extra, Bytes: AllocBytes> Allocation<Prov, Extra, Bytes> {
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn size(&self) -> Size {
        Size::from_bytes(self.len())
    }

    /// Looks at a slice which may contain uninitialized bytes or provenance. This differs
    /// from `get_bytes_with_uninit_and_ptr` in that it does no provenance checks (even on the
    /// edges) at all.
    /// This must not be used for reads affecting the interpreter execution.
    pub fn inspect_with_uninit_and_ptr_outside_interpreter(&self, range: Range<usize>) -> &[u8] {
        &self.bytes[range]
    }

    /// Returns the mask indicating which bytes are initialized.
    pub fn init_mask(&self) -> &InitMask {
        &self.init_mask
    }

    /// Returns the provenance map.
    pub fn provenance(&self) -> &ProvenanceMap<Prov> {
        &self.provenance
    }
}

/// Byte accessors.
impl<Prov: Provenance, Extra, Bytes: AllocBytes> Allocation<Prov, Extra, Bytes> {
    pub fn base_addr(&self) -> *const u8 {
        self.bytes.as_ptr()
    }

    /// This is the entirely abstraction-violating way to just grab the raw bytes without
    /// caring about provenance or initialization.
    ///
    /// This function also guarantees that the resulting pointer will remain stable
    /// even when new allocations are pushed to the `HashMap`. `mem_copy_repeatedly` relies
    /// on that.
    #[inline]
    pub fn get_bytes_unchecked(&self, range: AllocRange) -> &[u8] {
        &self.bytes[range.start.bytes_usize()..range.end().bytes_usize()]
    }

    /// Checks that these bytes are initialized, and then strip provenance (if possible) and return
    /// them.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    /// Most likely, you want to use the `PlaceTy` and `OperandTy`-based methods
    /// on `InterpCx` instead.
    #[inline]
    pub fn get_bytes_strip_provenance(
        &self,
        cx: &impl HasDataLayout,
        range: AllocRange,
    ) -> AllocResult<&[u8]> {
        self.init_mask.is_range_initialized(range).map_err(|uninit_range| {
            AllocError::InvalidUninitBytes(Some(UninitBytesAccess {
                access: range,
                uninit: uninit_range,
            }))
        })?;
        if !Prov::OFFSET_IS_ADDR {
            if !self.provenance.range_empty(range, cx) {
                return Err(AllocError::ReadPointerAsBytes);
            }
        }
        Ok(self.get_bytes_unchecked(range))
    }

    /// Just calling this already marks everything as defined and removes provenance,
    /// so be sure to actually put data there!
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    /// Most likely, you want to use the `PlaceTy` and `OperandTy`-based methods
    /// on `InterpCx` instead.
    pub fn get_bytes_mut(
        &mut self,
        cx: &impl HasDataLayout,
        range: AllocRange,
    ) -> AllocResult<&mut [u8]> {
        self.mark_init(range, true);
        self.provenance.clear(range, cx)?;

        Ok(&mut self.bytes[range.start.bytes_usize()..range.end().bytes_usize()])
    }

    /// A raw pointer variant of `get_bytes_mut` that avoids invalidating existing aliases into this memory.
    pub fn get_bytes_mut_ptr(
        &mut self,
        cx: &impl HasDataLayout,
        range: AllocRange,
    ) -> AllocResult<*mut [u8]> {
        self.mark_init(range, true);
        self.provenance.clear(range, cx)?;

        assert!(range.end().bytes_usize() <= self.bytes.len()); // need to do our own bounds-check
        let begin_ptr = self.bytes.as_mut_ptr().wrapping_add(range.start.bytes_usize());
        let len = range.end().bytes_usize() - range.start.bytes_usize();
        Ok(ptr::slice_from_raw_parts_mut(begin_ptr, len))
    }
}

/// Reading and writing.
impl<Prov: Provenance, Extra, Bytes: AllocBytes> Allocation<Prov, Extra, Bytes> {
    /// Sets the init bit for the given range.
    fn mark_init(&mut self, range: AllocRange, is_init: bool) {
        if range.size.bytes() == 0 {
            return;
        }
        assert!(self.mutability == Mutability::Mut);
        self.init_mask.set_range(range, is_init);
    }

    /// Reads a *non-ZST* scalar.
    ///
    /// If `read_provenance` is `true`, this will also read provenance; otherwise (if the machine
    /// supports that) provenance is entirely ignored.
    ///
    /// ZSTs can't be read because in order to obtain a `Pointer`, we need to check
    /// for ZSTness anyway due to integer pointers being valid for ZSTs.
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    /// Most likely, you want to call `InterpCx::read_scalar` instead of this method.
    pub fn read_scalar(
        &self,
        cx: &impl HasDataLayout,
        range: AllocRange,
        read_provenance: bool,
    ) -> AllocResult<Scalar<Prov>> {
        // First and foremost, if anything is uninit, bail.
        if self.init_mask.is_range_initialized(range).is_err() {
            return Err(AllocError::InvalidUninitBytes(None));
        }

        // Get the integer part of the result. We HAVE TO check provenance before returning this!
        let bytes = self.get_bytes_unchecked(range);
        let bits = read_target_uint(cx.data_layout().endian, bytes).unwrap();

        if read_provenance {
            assert_eq!(range.size, cx.data_layout().pointer_size);

            // When reading data with provenance, the easy case is finding provenance exactly where we
            // are reading, then we can put data and provenance back together and return that.
            if let Some(prov) = self.provenance.get_ptr(range.start) {
                // Now we can return the bits, with their appropriate provenance.
                let ptr = Pointer::new(prov, Size::from_bytes(bits));
                return Ok(Scalar::from_pointer(ptr, cx));
            }

            // If we can work on pointers byte-wise, join the byte-wise provenances.
            if Prov::OFFSET_IS_ADDR {
                let mut prov = self.provenance.get(range.start, cx);
                for offset in Size::from_bytes(1)..range.size {
                    let this_prov = self.provenance.get(range.start + offset, cx);
                    prov = Prov::join(prov, this_prov);
                }
                // Now use this provenance.
                let ptr = Pointer::new(prov, Size::from_bytes(bits));
                return Ok(Scalar::from_maybe_pointer(ptr, cx));
            }
        } else {
            // We are *not* reading a pointer.
            // If we can just ignore provenance, do exactly that.
            if Prov::OFFSET_IS_ADDR {
                // We just strip provenance.
                return Ok(Scalar::from_uint(bits, range.size));
            }
        }

        // Fallback path for when we cannot treat provenance bytewise or ignore it.
        assert!(!Prov::OFFSET_IS_ADDR);
        if !self.provenance.range_empty(range, cx) {
            return Err(AllocError::ReadPointerAsBytes);
        }
        // There is no provenance, we can just return the bits.
        Ok(Scalar::from_uint(bits, range.size))
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
        range: AllocRange,
        val: Scalar<Prov>,
    ) -> AllocResult {
        assert!(self.mutability == Mutability::Mut);

        // `to_bits_or_ptr_internal` is the right method because we just want to store this data
        // as-is into memory.
        let (bytes, provenance) = match val.to_bits_or_ptr_internal(range.size)? {
            Right(ptr) => {
                let (provenance, offset) = ptr.into_parts();
                (u128::from(offset.bytes()), Some(provenance))
            }
            Left(data) => (data, None),
        };

        let endian = cx.data_layout().endian;
        let dst = self.get_bytes_mut(cx, range)?;
        write_target_uint(endian, dst, bytes).unwrap();

        // See if we have to also store some provenance.
        if let Some(provenance) = provenance {
            assert_eq!(range.size, cx.data_layout().pointer_size);
            self.provenance.insert_ptr(range.start, provenance, cx);
        }

        Ok(())
    }

    /// Write "uninit" to the given memory range.
    pub fn write_uninit(&mut self, cx: &impl HasDataLayout, range: AllocRange) -> AllocResult {
        self.mark_init(range, false);
        self.provenance.clear(range, cx)?;
        return Ok(());
    }

    /// Applies a previously prepared provenance copy.
    /// The affected range, as defined in the parameters to `provenance().prepare_copy` is expected
    /// to be clear of provenance.
    ///
    /// This is dangerous to use as it can violate internal `Allocation` invariants!
    /// It only exists to support an efficient implementation of `mem_copy_repeatedly`.
    pub fn provenance_apply_copy(&mut self, copy: ProvenanceCopy<Prov>) {
        self.provenance.apply_copy(copy)
    }

    /// Applies a previously prepared copy of the init mask.
    ///
    /// This is dangerous to use as it can violate internal `Allocation` invariants!
    /// It only exists to support an efficient implementation of `mem_copy_repeatedly`.
    pub fn init_mask_apply_copy(&mut self, copy: InitCopy, range: AllocRange, repeat: u64) {
        self.init_mask.apply_copy(copy, range, repeat)
    }
}
