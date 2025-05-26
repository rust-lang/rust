//! The virtual memory representation of the MIR interpreter.

mod init_mask;
mod provenance_map;

use std::borrow::Cow;
use std::hash::Hash;
use std::ops::{Deref, DerefMut, Range};
use std::{fmt, hash, ptr};

use either::{Left, Right};
use init_mask::*;
pub use init_mask::{InitChunk, InitChunkIter};
use provenance_map::*;
use rustc_abi::{Align, HasDataLayout, Size};
use rustc_ast::Mutability;
use rustc_data_structures::intern::Interned;
use rustc_macros::HashStable;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

use super::{
    AllocId, BadBytesAccess, CtfeProvenance, InterpErrorKind, InterpResult, Pointer,
    PointerArithmetic, Provenance, ResourceExhaustionInfo, Scalar, ScalarSizeMismatch,
    UndefinedBehaviorInfo, UnsupportedOpInfo, interp_ok, read_target_uint, write_target_uint,
};
use crate::ty;

/// Functionality required for the bytes of an `Allocation`.
pub trait AllocBytes: Clone + fmt::Debug + Deref<Target = [u8]> + DerefMut<Target = [u8]> {
    /// The type of extra parameters passed in when creating an allocation.
    /// Can be used by `interpret::Machine` instances to make runtime-configuration-dependent
    /// decisions about the allocation strategy.
    type AllocParams;

    /// Create an `AllocBytes` from a slice of `u8`.
    fn from_bytes<'a>(
        slice: impl Into<Cow<'a, [u8]>>,
        _align: Align,
        _params: Self::AllocParams,
    ) -> Self;

    /// Create a zeroed `AllocBytes` of the specified size and alignment.
    /// Returns `None` if we ran out of memory on the host.
    fn zeroed(size: Size, _align: Align, _params: Self::AllocParams) -> Option<Self>;

    /// Gives direct access to the raw underlying storage.
    ///
    /// Crucially this pointer is compatible with:
    /// - other pointers returned by this method, and
    /// - references returned from `deref()`, as long as there was no write.
    fn as_mut_ptr(&mut self) -> *mut u8;

    /// Gives direct access to the raw underlying storage.
    ///
    /// Crucially this pointer is compatible with:
    /// - other pointers returned by this method, and
    /// - references returned from `deref()`, as long as there was no write.
    fn as_ptr(&self) -> *const u8;
}

/// Default `bytes` for `Allocation` is a `Box<u8>`.
impl AllocBytes for Box<[u8]> {
    type AllocParams = ();

    fn from_bytes<'a>(slice: impl Into<Cow<'a, [u8]>>, _align: Align, _params: ()) -> Self {
        Box::<[u8]>::from(slice.into())
    }

    fn zeroed(size: Size, _align: Align, _params: ()) -> Option<Self> {
        let bytes = Box::<[u8]>::try_new_zeroed_slice(size.bytes().try_into().ok()?).ok()?;
        // SAFETY: the box was zero-allocated, which is a valid initial value for Box<[u8]>
        let bytes = unsafe { bytes.assume_init() };
        Some(bytes)
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        Box::as_mut_ptr(self).cast()
    }

    fn as_ptr(&self) -> *const u8 {
        Box::as_ptr(self).cast()
    }
}

/// This type represents an Allocation in the Miri/CTFE core engine.
///
/// Its public API is rather low-level, working directly with allocation offsets and a custom error
/// type to account for the lack of an AllocId on this level. The Miri/CTFE core engine `memory`
/// module provides higher-level access.
// Note: for performance reasons when interning, some of the `Allocation` fields can be partially
// hashed. (see the `Hash` impl below for more details), so the impl is not derived.
#[derive(Clone, Eq, PartialEq)]
#[derive(HashStable)]
pub struct Allocation<Prov: Provenance = CtfeProvenance, Extra = (), Bytes = Box<[u8]>> {
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

/// Helper struct that packs an alignment, mutability, and "all bytes are zero" flag together.
///
/// Alignment values always have 2 free high bits, and we check for this in our [`Encodable`] impl.
struct AllocFlags {
    align: Align,
    mutability: Mutability,
    all_zero: bool,
}

impl<E: Encoder> Encodable<E> for AllocFlags {
    fn encode(&self, encoder: &mut E) {
        // Make sure Align::MAX can be stored with the high 2 bits unset.
        const {
            let max_supported_align_repr = u8::MAX >> 2;
            let max_supported_align = 1 << max_supported_align_repr;
            assert!(Align::MAX.bytes() <= max_supported_align)
        }

        let mut flags = self.align.bytes().trailing_zeros() as u8;
        flags |= match self.mutability {
            Mutability::Not => 0,
            Mutability::Mut => 1 << 6,
        };
        flags |= (self.all_zero as u8) << 7;
        flags.encode(encoder);
    }
}

impl<D: Decoder> Decodable<D> for AllocFlags {
    fn decode(decoder: &mut D) -> Self {
        let flags: u8 = Decodable::decode(decoder);
        let align = flags & 0b0011_1111;
        let mutability = flags & 0b0100_0000;
        let all_zero = flags & 0b1000_0000;

        let align = Align::from_bytes(1 << align).unwrap();
        let mutability = match mutability {
            0 => Mutability::Not,
            _ => Mutability::Mut,
        };
        let all_zero = all_zero > 0;

        AllocFlags { align, mutability, all_zero }
    }
}

/// Efficiently detect whether a slice of `u8` is all zero.
///
/// This is used in encoding of [`Allocation`] to special-case all-zero allocations. It is only
/// optimized a little, because for many allocations the encoding of the actual bytes does not
/// dominate runtime.
#[inline]
fn all_zero(buf: &[u8]) -> bool {
    // In the empty case we wouldn't encode any contents even without this system where we
    // special-case allocations whose contents are all 0. We can return anything in the empty case.
    if buf.is_empty() {
        return true;
    }
    // Just fast-rejecting based on the first element significantly reduces the amount that we end
    // up walking the whole array.
    if buf[0] != 0 {
        return false;
    }

    // This strategy of combining all slice elements with & or | is unbeatable for the large
    // all-zero case because it is so well-understood by autovectorization.
    buf.iter().fold(true, |acc, b| acc & (*b == 0))
}

/// Custom encoder for [`Allocation`] to more efficiently represent the case where all bytes are 0.
impl<Prov: Provenance, Extra, E: Encoder> Encodable<E> for Allocation<Prov, Extra, Box<[u8]>>
where
    ProvenanceMap<Prov>: Encodable<E>,
    Extra: Encodable<E>,
{
    fn encode(&self, encoder: &mut E) {
        let all_zero = all_zero(&self.bytes);
        AllocFlags { align: self.align, mutability: self.mutability, all_zero }.encode(encoder);

        encoder.emit_usize(self.bytes.len());
        if !all_zero {
            encoder.emit_raw_bytes(&self.bytes);
        }
        self.provenance.encode(encoder);
        self.init_mask.encode(encoder);
        self.extra.encode(encoder);
    }
}

impl<Prov: Provenance, Extra, D: Decoder> Decodable<D> for Allocation<Prov, Extra, Box<[u8]>>
where
    ProvenanceMap<Prov>: Decodable<D>,
    Extra: Decodable<D>,
{
    fn decode(decoder: &mut D) -> Self {
        let AllocFlags { align, mutability, all_zero } = Decodable::decode(decoder);

        let len = decoder.read_usize();
        let bytes = if all_zero { vec![0u8; len] } else { decoder.read_raw_bytes(len).to_vec() };
        let bytes = <Box<[u8]> as AllocBytes>::from_bytes(bytes, align, ());

        let provenance = Decodable::decode(decoder);
        let init_mask = Decodable::decode(decoder);
        let extra = Decodable::decode(decoder);

        Self { bytes, provenance, init_mask, align, mutability, extra }
    }
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
    ReadPointerAsInt(Option<BadBytesAccess>),
    /// Partially overwriting a pointer.
    OverwritePartialPointer(Size),
    /// Partially copying a pointer.
    ReadPartialPointer(Size),
    /// Using uninitialized data where it is not allowed.
    InvalidUninitBytes(Option<BadBytesAccess>),
}
pub type AllocResult<T = ()> = Result<T, AllocError>;

impl From<ScalarSizeMismatch> for AllocError {
    fn from(s: ScalarSizeMismatch) -> Self {
        AllocError::ScalarSizeMismatch(s)
    }
}

impl AllocError {
    pub fn to_interp_error<'tcx>(self, alloc_id: AllocId) -> InterpErrorKind<'tcx> {
        use AllocError::*;
        match self {
            ScalarSizeMismatch(s) => {
                InterpErrorKind::UndefinedBehavior(UndefinedBehaviorInfo::ScalarSizeMismatch(s))
            }
            ReadPointerAsInt(info) => InterpErrorKind::Unsupported(
                UnsupportedOpInfo::ReadPointerAsInt(info.map(|b| (alloc_id, b))),
            ),
            OverwritePartialPointer(offset) => InterpErrorKind::Unsupported(
                UnsupportedOpInfo::OverwritePartialPointer(Pointer::new(alloc_id, offset)),
            ),
            ReadPartialPointer(offset) => InterpErrorKind::Unsupported(
                UnsupportedOpInfo::ReadPartialPointer(Pointer::new(alloc_id, offset)),
            ),
            InvalidUninitBytes(info) => InterpErrorKind::UndefinedBehavior(
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

/// Whether a new allocation should be initialized with zero-bytes.
pub enum AllocInit {
    Uninit,
    Zero,
}

// The constructors are all without extra; the extra gets added by a machine hook later.
impl<Prov: Provenance, Bytes: AllocBytes> Allocation<Prov, (), Bytes> {
    /// Creates an allocation initialized by the given bytes
    pub fn from_bytes<'a>(
        slice: impl Into<Cow<'a, [u8]>>,
        align: Align,
        mutability: Mutability,
        params: <Bytes as AllocBytes>::AllocParams,
    ) -> Self {
        let bytes = Bytes::from_bytes(slice, align, params);
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

    pub fn from_bytes_byte_aligned_immutable<'a>(
        slice: impl Into<Cow<'a, [u8]>>,
        params: <Bytes as AllocBytes>::AllocParams,
    ) -> Self {
        Allocation::from_bytes(slice, Align::ONE, Mutability::Not, params)
    }

    fn new_inner<R>(
        size: Size,
        align: Align,
        init: AllocInit,
        params: <Bytes as AllocBytes>::AllocParams,
        fail: impl FnOnce() -> R,
    ) -> Result<Self, R> {
        // We raise an error if we cannot create the allocation on the host.
        // This results in an error that can happen non-deterministically, since the memory
        // available to the compiler can change between runs. Normally queries are always
        // deterministic. However, we can be non-deterministic here because all uses of const
        // evaluation (including ConstProp!) will make compilation fail (via hard error
        // or ICE) upon encountering a `MemoryExhausted` error.
        let bytes = Bytes::zeroed(size, align, params).ok_or_else(fail)?;

        Ok(Allocation {
            bytes,
            provenance: ProvenanceMap::new(),
            init_mask: InitMask::new(
                size,
                match init {
                    AllocInit::Uninit => false,
                    AllocInit::Zero => true,
                },
            ),
            align,
            mutability: Mutability::Mut,
            extra: (),
        })
    }

    /// Try to create an Allocation of `size` bytes, failing if there is not enough memory
    /// available to the compiler to do so.
    pub fn try_new<'tcx>(
        size: Size,
        align: Align,
        init: AllocInit,
        params: <Bytes as AllocBytes>::AllocParams,
    ) -> InterpResult<'tcx, Self> {
        Self::new_inner(size, align, init, params, || {
            ty::tls::with(|tcx| tcx.dcx().delayed_bug("exhausted memory during interpretation"));
            InterpErrorKind::ResourceExhaustion(ResourceExhaustionInfo::MemoryExhausted)
        })
        .into()
    }

    /// Try to create an Allocation of `size` bytes, panics if there is not enough memory
    /// available to the compiler to do so.
    ///
    /// Example use case: To obtain an Allocation filled with specific data,
    /// first call this function and then call write_scalar to fill in the right data.
    pub fn new(
        size: Size,
        align: Align,
        init: AllocInit,
        params: <Bytes as AllocBytes>::AllocParams,
    ) -> Self {
        match Self::new_inner(size, align, init, params, || {
            panic!(
                "interpreter ran out of memory: cannot create allocation of {} bytes",
                size.bytes()
            );
        }) {
            Ok(x) => x,
            Err(x) => x,
        }
    }

    /// Add the extra.
    pub fn with_extra<Extra>(self, extra: Extra) -> Allocation<Prov, Extra, Bytes> {
        Allocation {
            bytes: self.bytes,
            provenance: self.provenance,
            init_mask: self.init_mask,
            align: self.align,
            mutability: self.mutability,
            extra,
        }
    }
}

impl Allocation {
    /// Adjust allocation from the ones in `tcx` to a custom Machine instance
    /// with a different `Provenance` and `Byte` type.
    pub fn adjust_from_tcx<'tcx, Prov: Provenance, Bytes: AllocBytes>(
        &self,
        cx: &impl HasDataLayout,
        mut alloc_bytes: impl FnMut(&[u8], Align) -> InterpResult<'tcx, Bytes>,
        mut adjust_ptr: impl FnMut(Pointer<CtfeProvenance>) -> InterpResult<'tcx, Pointer<Prov>>,
    ) -> InterpResult<'tcx, Allocation<Prov, (), Bytes>> {
        // Copy the data.
        let mut bytes = alloc_bytes(&*self.bytes, self.align)?;
        // Adjust provenance of pointers stored in this allocation.
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
        interp_ok(Allocation {
            bytes,
            provenance: ProvenanceMap::from_presorted_ptrs(new_provenance),
            init_mask: self.init_mask.clone(),
            align: self.align,
            mutability: self.mutability,
            extra: self.extra,
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
            AllocError::InvalidUninitBytes(Some(BadBytesAccess {
                access: range,
                bad: uninit_range,
            }))
        })?;
        if !Prov::OFFSET_IS_ADDR && !self.provenance.range_empty(range, cx) {
            // Find the provenance.
            let (offset, _prov) = self
                .provenance
                .range_ptrs_get(range, cx)
                .first()
                .copied()
                .expect("there must be provenance somewhere here");
            let start = offset.max(range.start); // the pointer might begin before `range`!
            let end = (offset + cx.pointer_size()).min(range.end()); // the pointer might end after `range`!
            return Err(AllocError::ReadPointerAsInt(Some(BadBytesAccess {
                access: range,
                bad: AllocRange::from(start..end),
            })));
        }
        Ok(self.get_bytes_unchecked(range))
    }

    /// This is the entirely abstraction-violating way to just get mutable access to the raw bytes.
    /// Just calling this already marks everything as defined and removes provenance, so be sure to
    /// actually overwrite all the data there!
    ///
    /// It is the caller's responsibility to check bounds and alignment beforehand.
    /// Most likely, you want to use the `PlaceTy` and `OperandTy`-based methods
    /// on `InterpCx` instead.
    pub fn get_bytes_unchecked_for_overwrite(
        &mut self,
        cx: &impl HasDataLayout,
        range: AllocRange,
    ) -> AllocResult<&mut [u8]> {
        self.mark_init(range, true);
        self.provenance.clear(range, cx)?;

        Ok(&mut self.bytes[range.start.bytes_usize()..range.end().bytes_usize()])
    }

    /// A raw pointer variant of `get_bytes_unchecked_for_overwrite` that avoids invalidating existing immutable aliases
    /// into this memory.
    pub fn get_bytes_unchecked_for_overwrite_ptr(
        &mut self,
        cx: &impl HasDataLayout,
        range: AllocRange,
    ) -> AllocResult<*mut [u8]> {
        self.mark_init(range, true);
        self.provenance.clear(range, cx)?;

        assert!(range.end().bytes_usize() <= self.bytes.len()); // need to do our own bounds-check
        // Crucially, we go via `AllocBytes::as_mut_ptr`, not `AllocBytes::deref_mut`.
        let begin_ptr = self.bytes.as_mut_ptr().wrapping_add(range.start.bytes_usize());
        let len = range.end().bytes_usize() - range.start.bytes_usize();
        Ok(ptr::slice_from_raw_parts_mut(begin_ptr, len))
    }

    /// This gives direct mutable access to the entire buffer, just exposing their internal state
    /// without resetting anything. Directly exposes `AllocBytes::as_mut_ptr`. Only works if
    /// `OFFSET_IS_ADDR` is true.
    pub fn get_bytes_unchecked_raw_mut(&mut self) -> *mut u8 {
        assert!(Prov::OFFSET_IS_ADDR);
        self.bytes.as_mut_ptr()
    }

    /// This gives direct immutable access to the entire buffer, just exposing their internal state
    /// without resetting anything. Directly exposes `AllocBytes::as_ptr`. Only works if
    /// `OFFSET_IS_ADDR` is true.
    pub fn get_bytes_unchecked_raw(&self) -> *const u8 {
        assert!(Prov::OFFSET_IS_ADDR);
        self.bytes.as_ptr()
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
            } else {
                // Without OFFSET_IS_ADDR, the only remaining case we can handle is total absence of
                // provenance.
                if self.provenance.range_empty(range, cx) {
                    return Ok(Scalar::from_uint(bits, range.size));
                }
                // Else we have mixed provenance, that doesn't work.
                return Err(AllocError::ReadPartialPointer(range.start));
            }
        } else {
            // We are *not* reading a pointer.
            // If we can just ignore provenance or there is none, that's easy.
            if Prov::OFFSET_IS_ADDR || self.provenance.range_empty(range, cx) {
                // We just strip provenance.
                return Ok(Scalar::from_uint(bits, range.size));
            }
            // There is some provenance and we don't have OFFSET_IS_ADDR. This doesn't work.
            return Err(AllocError::ReadPointerAsInt(None));
        }
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
        // as-is into memory. This also double-checks that `val.size()` matches `range.size`.
        let (bytes, provenance) = match val.to_bits_or_ptr_internal(range.size)? {
            Right(ptr) => {
                let (provenance, offset) = ptr.into_parts();
                (u128::from(offset.bytes()), Some(provenance))
            }
            Left(data) => (data, None),
        };

        let endian = cx.data_layout().endian;
        // Yes we do overwrite all the bytes in `dst`.
        let dst = self.get_bytes_unchecked_for_overwrite(cx, range)?;
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
        Ok(())
    }

    /// Initialize all previously uninitialized bytes in the entire allocation, and set
    /// provenance of everything to `Wildcard`. Before calling this, make sure all
    /// provenance in this allocation is exposed!
    pub fn prepare_for_native_write(&mut self) -> AllocResult {
        let full_range = AllocRange { start: Size::ZERO, size: Size::from_bytes(self.len()) };
        // Overwrite uninitialized bytes with 0, to ensure we don't leak whatever their value happens to be.
        for chunk in self.init_mask.range_as_init_chunks(full_range) {
            if !chunk.is_init() {
                let uninit_bytes = &mut self.bytes
                    [chunk.range().start.bytes_usize()..chunk.range().end.bytes_usize()];
                uninit_bytes.fill(0);
            }
        }
        // Mark everything as initialized now.
        self.mark_init(full_range, true);

        // Set provenance of all bytes to wildcard.
        self.provenance.write_wildcards(self.len());

        // Also expose the provenance of the interpreter-level allocation, so it can
        // be written by FFI. The `black_box` is defensive programming as LLVM likes
        // to (incorrectly) optimize away ptr2int casts whose result is unused.
        std::hint::black_box(self.get_bytes_unchecked_raw_mut().expose_provenance());

        Ok(())
    }

    /// Remove all provenance in the given memory range.
    pub fn clear_provenance(&mut self, cx: &impl HasDataLayout, range: AllocRange) -> AllocResult {
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
