//! Store the provenance for each byte in the range, with a more efficient
//! representation for the common case where PTR_SIZE consecutive bytes have the same provenance.

use std::cmp;
use std::ops::Range;

use rustc_abi::{HasDataLayout, Size};
use rustc_data_structures::sorted_map::SortedMap;
use rustc_macros::HashStable;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use tracing::trace;

use super::{AllocError, AllocRange, AllocResult, CtfeProvenance, Provenance, alloc_range};

/// Stores the provenance information of pointers stored in memory.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[derive(HashStable)]
pub struct ProvenanceMap<Prov = CtfeProvenance> {
    /// `Provenance` in this map applies from the given offset for an entire pointer-size worth of
    /// bytes. Two entries in this map are always at least a pointer size apart.
    ptrs: SortedMap<Size, Prov>,
    /// Provenance in this map only applies to the given single byte.
    /// This map is disjoint from the previous. It will always be empty when
    /// `Prov::OFFSET_IS_ADDR` is false.
    bytes: Option<Box<SortedMap<Size, Prov>>>,
}

// These impls are generic over `Prov` since `CtfeProvenance` is only decodable/encodable
// for some particular `D`/`S`.
impl<D: Decoder, Prov: Provenance + Decodable<D>> Decodable<D> for ProvenanceMap<Prov> {
    fn decode(d: &mut D) -> Self {
        assert!(!Prov::OFFSET_IS_ADDR); // only `CtfeProvenance` is ever serialized
        Self { ptrs: Decodable::decode(d), bytes: None }
    }
}
impl<S: Encoder, Prov: Provenance + Encodable<S>> Encodable<S> for ProvenanceMap<Prov> {
    fn encode(&self, s: &mut S) {
        let Self { ptrs, bytes } = self;
        assert!(!Prov::OFFSET_IS_ADDR); // only `CtfeProvenance` is ever serialized
        debug_assert!(bytes.is_none()); // without `OFFSET_IS_ADDR`, this is always empty
        ptrs.encode(s)
    }
}

impl<Prov> ProvenanceMap<Prov> {
    pub fn new() -> Self {
        ProvenanceMap { ptrs: SortedMap::new(), bytes: None }
    }

    /// The caller must guarantee that the given provenance list is already sorted
    /// by address and contain no duplicates.
    pub fn from_presorted_ptrs(r: Vec<(Size, Prov)>) -> Self {
        ProvenanceMap { ptrs: SortedMap::from_presorted_elements(r), bytes: None }
    }
}

impl ProvenanceMap {
    /// Give access to the ptr-sized provenances (which can also be thought of as relocations, and
    /// indeed that is how codegen treats them).
    ///
    /// Only exposed with `CtfeProvenance` provenance, since it panics if there is bytewise provenance.
    #[inline]
    pub fn ptrs(&self) -> &SortedMap<Size, CtfeProvenance> {
        debug_assert!(self.bytes.is_none()); // `CtfeProvenance::OFFSET_IS_ADDR` is false so this cannot fail
        &self.ptrs
    }
}

impl<Prov: Provenance> ProvenanceMap<Prov> {
    fn adjusted_range_ptrs(range: AllocRange, cx: &impl HasDataLayout) -> Range<Size> {
        // We have to go back `pointer_size - 1` bytes, as that one would still overlap with
        // the beginning of this range.
        let adjusted_start = Size::from_bytes(
            range.start.bytes().saturating_sub(cx.data_layout().pointer_size.bytes() - 1),
        );
        adjusted_start..range.end()
    }

    /// Returns all ptr-sized provenance in the given range.
    /// If the range has length 0, returns provenance that crosses the edge between `start-1` and
    /// `start`.
    pub(super) fn range_ptrs_get(
        &self,
        range: AllocRange,
        cx: &impl HasDataLayout,
    ) -> &[(Size, Prov)] {
        self.ptrs.range(Self::adjusted_range_ptrs(range, cx))
    }

    /// `pm.range_ptrs_is_empty(r, cx)` == `pm.range_ptrs_get(r, cx).is_empty()`, but is faster.
    pub(super) fn range_ptrs_is_empty(&self, range: AllocRange, cx: &impl HasDataLayout) -> bool {
        self.ptrs.range_is_empty(Self::adjusted_range_ptrs(range, cx))
    }

    /// Returns all byte-wise provenance in the given range.
    fn range_bytes_get(&self, range: AllocRange) -> &[(Size, Prov)] {
        if let Some(bytes) = self.bytes.as_ref() {
            bytes.range(range.start..range.end())
        } else {
            &[]
        }
    }

    /// Same as `range_bytes_get(range).is_empty()`, but faster.
    fn range_bytes_is_empty(&self, range: AllocRange) -> bool {
        self.bytes.as_ref().is_none_or(|bytes| bytes.range_is_empty(range.start..range.end()))
    }

    /// Get the provenance of a single byte.
    pub fn get(&self, offset: Size, cx: &impl HasDataLayout) -> Option<Prov> {
        let prov = self.range_ptrs_get(alloc_range(offset, Size::from_bytes(1)), cx);
        debug_assert!(prov.len() <= 1);
        if let Some(entry) = prov.first() {
            // If it overlaps with this byte, it is on this byte.
            debug_assert!(self.bytes.as_ref().is_none_or(|b| !b.contains_key(&offset)));
            Some(entry.1)
        } else {
            // Look up per-byte provenance.
            self.bytes.as_ref().and_then(|b| b.get(&offset).copied())
        }
    }

    /// Check if here is ptr-sized provenance at the given index.
    /// Does not mean anything for bytewise provenance! But can be useful as an optimization.
    pub fn get_ptr(&self, offset: Size) -> Option<Prov> {
        self.ptrs.get(&offset).copied()
    }

    /// Returns whether this allocation has provenance overlapping with the given range.
    ///
    /// Note: this function exists to allow `range_get_provenance` to be private, in order to somewhat
    /// limit access to provenance outside of the `Allocation` abstraction.
    ///
    pub fn range_empty(&self, range: AllocRange, cx: &impl HasDataLayout) -> bool {
        self.range_ptrs_is_empty(range, cx) && self.range_bytes_is_empty(range)
    }

    /// Yields all the provenances stored in this map.
    pub fn provenances(&self) -> impl Iterator<Item = Prov> {
        let bytes = self.bytes.iter().flat_map(|b| b.values());
        self.ptrs.values().chain(bytes).copied()
    }

    pub fn insert_ptr(&mut self, offset: Size, prov: Prov, cx: &impl HasDataLayout) {
        debug_assert!(self.range_empty(alloc_range(offset, cx.data_layout().pointer_size), cx));
        self.ptrs.insert(offset, prov);
    }

    /// Removes all provenance inside the given range.
    /// If there is provenance overlapping with the edges, might result in an error.
    pub fn clear(&mut self, range: AllocRange, cx: &impl HasDataLayout) -> AllocResult {
        let start = range.start;
        let end = range.end();
        // Clear the bytewise part -- this is easy.
        if Prov::OFFSET_IS_ADDR {
            if let Some(bytes) = self.bytes.as_mut() {
                bytes.remove_range(start..end);
            }
        } else {
            debug_assert!(self.bytes.is_none());
        }

        // For the ptr-sized part, find the first (inclusive) and last (exclusive) byte of
        // provenance that overlaps with the given range.
        let (first, last) = {
            // Find all provenance overlapping the given range.
            if self.range_ptrs_is_empty(range, cx) {
                // No provenance in this range, we are done. This is the common case.
                return Ok(());
            }

            // This redoes some of the work of `range_get_ptrs_is_empty`, but this path is much
            // colder than the early return above, so it's worth it.
            let provenance = self.range_ptrs_get(range, cx);
            (
                provenance.first().unwrap().0,
                provenance.last().unwrap().0 + cx.data_layout().pointer_size,
            )
        };

        // We need to handle clearing the provenance from parts of a pointer.
        if first < start {
            if !Prov::OFFSET_IS_ADDR {
                // We can't split up the provenance into less than a pointer.
                return Err(AllocError::OverwritePartialPointer(first));
            }
            // Insert the remaining part in the bytewise provenance.
            let prov = self.ptrs[&first];
            let bytes = self.bytes.get_or_insert_with(Box::default);
            for offset in first..start {
                bytes.insert(offset, prov);
            }
        }
        if last > end {
            let begin_of_last = last - cx.data_layout().pointer_size;
            if !Prov::OFFSET_IS_ADDR {
                // We can't split up the provenance into less than a pointer.
                return Err(AllocError::OverwritePartialPointer(begin_of_last));
            }
            // Insert the remaining part in the bytewise provenance.
            let prov = self.ptrs[&begin_of_last];
            let bytes = self.bytes.get_or_insert_with(Box::default);
            for offset in end..last {
                bytes.insert(offset, prov);
            }
        }

        // Forget all the provenance.
        // Since provenance do not overlap, we know that removing until `last` (exclusive) is fine,
        // i.e., this will not remove any other provenance just after the ones we care about.
        self.ptrs.remove_range(first..last);

        Ok(())
    }

    /// Overwrites all provenance in the allocation with wildcard provenance.
    ///
    /// Provided for usage in Miri and panics otherwise.
    pub fn write_wildcards(&mut self, alloc_size: usize) {
        assert!(
            Prov::OFFSET_IS_ADDR,
            "writing wildcard provenance is not supported when `OFFSET_IS_ADDR` is false"
        );
        let wildcard = Prov::WILDCARD.unwrap();

        // Remove all pointer provenances, then write wildcards into the whole byte range.
        self.ptrs.clear();
        let last = Size::from_bytes(alloc_size);
        let bytes = self.bytes.get_or_insert_with(Box::default);
        for offset in Size::ZERO..last {
            bytes.insert(offset, wildcard);
        }
    }
}

/// A partial, owned list of provenance to transfer into another allocation.
///
/// Offsets are already adjusted to the destination allocation.
pub struct ProvenanceCopy<Prov> {
    dest_ptrs: Option<Box<[(Size, Prov)]>>,
    dest_bytes: Option<Box<[(Size, Prov)]>>,
}

impl<Prov: Provenance> ProvenanceMap<Prov> {
    pub fn prepare_copy(
        &self,
        src: AllocRange,
        dest: Size,
        count: u64,
        cx: &impl HasDataLayout,
    ) -> AllocResult<ProvenanceCopy<Prov>> {
        let shift_offset = move |idx, offset| {
            // compute offset for current repetition
            let dest_offset = dest + src.size * idx; // `Size` operations
            // shift offsets from source allocation to destination allocation
            (offset - src.start) + dest_offset // `Size` operations
        };
        let ptr_size = cx.data_layout().pointer_size;

        // # Pointer-sized provenances
        // Get the provenances that are entirely within this range.
        // (Different from `range_get_ptrs` which asks if they overlap the range.)
        // Only makes sense if we are copying at least one pointer worth of bytes.
        let mut dest_ptrs_box = None;
        if src.size >= ptr_size {
            let adjusted_end = Size::from_bytes(src.end().bytes() - (ptr_size.bytes() - 1));
            let ptrs = self.ptrs.range(src.start..adjusted_end);
            // If `count` is large, this is rather wasteful -- we are allocating a big array here, which
            // is mostly filled with redundant information since it's just N copies of the same `Prov`s
            // at slightly adjusted offsets. The reason we do this is so that in `mark_provenance_range`
            // we can use `insert_presorted`. That wouldn't work with an `Iterator` that just produces
            // the right sequence of provenance for all N copies.
            // Basically, this large array would have to be created anyway in the target allocation.
            let mut dest_ptrs = Vec::with_capacity(ptrs.len() * (count as usize));
            for i in 0..count {
                dest_ptrs
                    .extend(ptrs.iter().map(|&(offset, reloc)| (shift_offset(i, offset), reloc)));
            }
            debug_assert_eq!(dest_ptrs.len(), dest_ptrs.capacity());
            dest_ptrs_box = Some(dest_ptrs.into_boxed_slice());
        };

        // # Byte-sized provenances
        // This includes the existing bytewise provenance in the range, and ptr provenance
        // that overlaps with the begin/end of the range.
        let mut dest_bytes_box = None;
        let begin_overlap = self.range_ptrs_get(alloc_range(src.start, Size::ZERO), cx).first();
        let end_overlap = self.range_ptrs_get(alloc_range(src.end(), Size::ZERO), cx).first();
        if !Prov::OFFSET_IS_ADDR {
            // There can't be any bytewise provenance, and we cannot split up the begin/end overlap.
            if let Some(entry) = begin_overlap {
                return Err(AllocError::ReadPartialPointer(entry.0));
            }
            if let Some(entry) = end_overlap {
                return Err(AllocError::ReadPartialPointer(entry.0));
            }
            debug_assert!(self.bytes.is_none());
        } else {
            let mut bytes = Vec::new();
            // First, if there is a part of a pointer at the start, add that.
            if let Some(entry) = begin_overlap {
                trace!("start overlapping entry: {entry:?}");
                // For really small copies, make sure we don't run off the end of the `src` range.
                let entry_end = cmp::min(entry.0 + ptr_size, src.end());
                for offset in src.start..entry_end {
                    bytes.push((offset, entry.1));
                }
            } else {
                trace!("no start overlapping entry");
            }

            // Then the main part, bytewise provenance from `self.bytes`.
            bytes.extend(self.range_bytes_get(src));

            // And finally possibly parts of a pointer at the end.
            if let Some(entry) = end_overlap {
                trace!("end overlapping entry: {entry:?}");
                // For really small copies, make sure we don't start before `src` does.
                let entry_start = cmp::max(entry.0, src.start);
                for offset in entry_start..src.end() {
                    if bytes.last().is_none_or(|bytes_entry| bytes_entry.0 < offset) {
                        // The last entry, if it exists, has a lower offset than us.
                        bytes.push((offset, entry.1));
                    } else {
                        // There already is an entry for this offset in there! This can happen when the
                        // start and end range checks actually end up hitting the same pointer, so we
                        // already added this in the "pointer at the start" part above.
                        assert!(entry.0 <= src.start);
                    }
                }
            } else {
                trace!("no end overlapping entry");
            }
            trace!("byte provenances: {bytes:?}");

            // And again a buffer for the new list on the target side.
            let mut dest_bytes = Vec::with_capacity(bytes.len() * (count as usize));
            for i in 0..count {
                dest_bytes
                    .extend(bytes.iter().map(|&(offset, reloc)| (shift_offset(i, offset), reloc)));
            }
            debug_assert_eq!(dest_bytes.len(), dest_bytes.capacity());
            dest_bytes_box = Some(dest_bytes.into_boxed_slice());
        }

        Ok(ProvenanceCopy { dest_ptrs: dest_ptrs_box, dest_bytes: dest_bytes_box })
    }

    /// Applies a provenance copy.
    /// The affected range, as defined in the parameters to `prepare_copy` is expected
    /// to be clear of provenance.
    pub fn apply_copy(&mut self, copy: ProvenanceCopy<Prov>) {
        if let Some(dest_ptrs) = copy.dest_ptrs {
            self.ptrs.insert_presorted(dest_ptrs.into());
        }
        if Prov::OFFSET_IS_ADDR {
            if let Some(dest_bytes) = copy.dest_bytes
                && !dest_bytes.is_empty()
            {
                self.bytes.get_or_insert_with(Box::default).insert_presorted(dest_bytes.into());
            }
        } else {
            debug_assert!(copy.dest_bytes.is_none());
        }
    }
}
