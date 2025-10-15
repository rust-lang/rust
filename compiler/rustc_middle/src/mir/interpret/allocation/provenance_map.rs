//! Store the provenance for each byte in the range, with a more efficient
//! representation for the common case where PTR_SIZE consecutive bytes have the same provenance.

use std::cmp;
use std::ops::Range;

use rustc_abi::{HasDataLayout, Size};
use rustc_data_structures::sorted_map::SortedMap;
use rustc_macros::HashStable;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use tracing::trace;

use super::{AllocRange, CtfeProvenance, Provenance, alloc_range};
use crate::mir::interpret::{AllocError, AllocResult};

/// Stores the provenance information of pointers stored in memory.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[derive(HashStable)]
pub struct ProvenanceMap<Prov = CtfeProvenance> {
    /// `Provenance` in this map applies from the given offset for an entire pointer-size worth of
    /// bytes. Two entries in this map are always at least a pointer size apart.
    ptrs: SortedMap<Size, Prov>,
    /// This stores byte-sized provenance fragments.
    /// The `u8` indicates the position of this byte inside its original pointer.
    /// If the bytes are re-assembled in their original order, the pointer can be used again.
    /// Wildcard provenance is allowed to have index 0 everywhere.
    bytes: Option<Box<SortedMap<Size, (Prov, u8)>>>,
}

// These impls are generic over `Prov` since `CtfeProvenance` is only decodable/encodable
// for some particular `D`/`S`.
impl<D: Decoder, Prov: Provenance + Decodable<D>> Decodable<D> for ProvenanceMap<Prov> {
    fn decode(d: &mut D) -> Self {
        // `bytes` is not in the serialized format
        Self { ptrs: Decodable::decode(d), bytes: None }
    }
}
impl<S: Encoder, Prov: Provenance + Encodable<S>> Encodable<S> for ProvenanceMap<Prov> {
    fn encode(&self, s: &mut S) {
        let Self { ptrs, bytes } = self;
        assert!(bytes.is_none()); // interning refuses allocations with pointer fragments
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
    /// Only use on interned allocations, as other allocations may have per-byte provenance!
    #[inline]
    pub fn ptrs(&self) -> &SortedMap<Size, CtfeProvenance> {
        assert!(self.bytes.is_none(), "`ptrs()` called on non-interned allocation");
        &self.ptrs
    }
}

impl<Prov: Provenance> ProvenanceMap<Prov> {
    fn adjusted_range_ptrs(range: AllocRange, cx: &impl HasDataLayout) -> Range<Size> {
        // We have to go back `pointer_size - 1` bytes, as that one would still overlap with
        // the beginning of this range.
        let adjusted_start = Size::from_bytes(
            range.start.bytes().saturating_sub(cx.data_layout().pointer_size().bytes() - 1),
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
    fn range_ptrs_is_empty(&self, range: AllocRange, cx: &impl HasDataLayout) -> bool {
        self.ptrs.range_is_empty(Self::adjusted_range_ptrs(range, cx))
    }

    /// Returns all byte-wise provenance in the given range.
    fn range_bytes_get(&self, range: AllocRange) -> &[(Size, (Prov, u8))] {
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
    pub fn get_byte(&self, offset: Size, cx: &impl HasDataLayout) -> Option<(Prov, u8)> {
        let prov = self.range_ptrs_get(alloc_range(offset, Size::from_bytes(1)), cx);
        debug_assert!(prov.len() <= 1);
        if let Some(entry) = prov.first() {
            // If it overlaps with this byte, it is on this byte.
            debug_assert!(self.bytes.as_ref().is_none_or(|b| !b.contains_key(&offset)));
            Some((entry.1, (offset - entry.0).bytes() as u8))
        } else {
            // Look up per-byte provenance.
            self.bytes.as_ref().and_then(|b| b.get(&offset).copied())
        }
    }

    /// Gets the provenances of all bytes (including from pointers) in a range.
    pub fn get_range(
        &self,
        cx: &impl HasDataLayout,
        range: AllocRange,
    ) -> impl Iterator<Item = Prov> {
        let ptr_provs = self.range_ptrs_get(range, cx).iter().map(|(_, p)| *p);
        let byte_provs = self.range_bytes_get(range).iter().map(|(_, (p, _))| *p);
        ptr_provs.chain(byte_provs)
    }

    /// Attempt to merge per-byte provenance back into ptr chunks, if the right fragments
    /// sit next to each other. Return `false` is that is not possible due to partial pointers.
    pub fn merge_bytes(&mut self, cx: &impl HasDataLayout) -> bool {
        let Some(bytes) = self.bytes.as_deref_mut() else {
            return true;
        };
        if !Prov::OFFSET_IS_ADDR {
            // FIXME(#146291): We need to ensure that we don't mix different pointers with
            // the same provenance.
            return false;
        }
        let ptr_size = cx.data_layout().pointer_size();
        while let Some((offset, (prov, _))) = bytes.iter().next().copied() {
            // Check if this fragment starts a pointer.
            let range = offset..offset + ptr_size;
            let frags = bytes.range(range.clone());
            if frags.len() != ptr_size.bytes_usize() {
                return false;
            }
            for (idx, (_offset, (frag_prov, frag_idx))) in frags.iter().copied().enumerate() {
                if frag_prov != prov || frag_idx != idx as u8 {
                    return false;
                }
            }
            // Looks like a pointer! Move it over to the ptr provenance map.
            bytes.remove_range(range);
            self.ptrs.insert(offset, prov);
        }
        // We managed to convert everything into whole pointers.
        self.bytes = None;
        true
    }

    /// Check if there is ptr-sized provenance at the given index.
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
        let bytes = self.bytes.iter().flat_map(|b| b.values().map(|(p, _i)| p));
        self.ptrs.values().chain(bytes).copied()
    }

    pub fn insert_ptr(&mut self, offset: Size, prov: Prov, cx: &impl HasDataLayout) {
        debug_assert!(self.range_empty(alloc_range(offset, cx.data_layout().pointer_size()), cx));
        self.ptrs.insert(offset, prov);
    }

    /// Removes all provenance inside the given range.
    /// If there is provenance overlapping with the edges, might result in an error.
    pub fn clear(&mut self, range: AllocRange, cx: &impl HasDataLayout) {
        let start = range.start;
        let end = range.end();
        // Clear the bytewise part -- this is easy.
        if let Some(bytes) = self.bytes.as_mut() {
            bytes.remove_range(start..end);
        }

        let pointer_size = cx.data_layout().pointer_size();

        // For the ptr-sized part, find the first (inclusive) and last (exclusive) byte of
        // provenance that overlaps with the given range.
        let (first, last) = {
            // Find all provenance overlapping the given range.
            if self.range_ptrs_is_empty(range, cx) {
                // No provenance in this range, we are done. This is the common case.
                return;
            }

            // This redoes some of the work of `range_get_ptrs_is_empty`, but this path is much
            // colder than the early return above, so it's worth it.
            let provenance = self.range_ptrs_get(range, cx);
            (provenance.first().unwrap().0, provenance.last().unwrap().0 + pointer_size)
        };

        // We need to handle clearing the provenance from parts of a pointer.
        if first < start {
            // Insert the remaining part in the bytewise provenance.
            let prov = self.ptrs[&first];
            let bytes = self.bytes.get_or_insert_with(Box::default);
            for offset in first..start {
                bytes.insert(offset, (prov, (offset - first).bytes() as u8));
            }
        }
        if last > end {
            let begin_of_last = last - pointer_size;
            // Insert the remaining part in the bytewise provenance.
            let prov = self.ptrs[&begin_of_last];
            let bytes = self.bytes.get_or_insert_with(Box::default);
            for offset in end..last {
                bytes.insert(offset, (prov, (offset - begin_of_last).bytes() as u8));
            }
        }

        // Forget all the provenance.
        // Since provenance do not overlap, we know that removing until `last` (exclusive) is fine,
        // i.e., this will not remove any other provenance just after the ones we care about.
        self.ptrs.remove_range(first..last);
    }

    /// Overwrites all provenance in the given range with wildcard provenance.
    /// Pointers partially overwritten will have their provenances preserved
    /// bytewise on their remaining bytes.
    ///
    /// Provided for usage in Miri and panics otherwise.
    pub fn write_wildcards(&mut self, cx: &impl HasDataLayout, range: AllocRange) {
        let wildcard = Prov::WILDCARD.unwrap();

        let bytes = self.bytes.get_or_insert_with(Box::default);

        // Remove pointer provenances that overlap with the range, then readd the edge ones bytewise.
        let ptr_range = Self::adjusted_range_ptrs(range, cx);
        let ptrs = self.ptrs.range(ptr_range.clone());
        if let Some((offset, prov)) = ptrs.first().copied() {
            for byte_ofs in offset..range.start {
                bytes.insert(byte_ofs, (prov, (byte_ofs - offset).bytes() as u8));
            }
        }
        if let Some((offset, prov)) = ptrs.last().copied() {
            for byte_ofs in range.end()..offset + cx.data_layout().pointer_size() {
                bytes.insert(byte_ofs, (prov, (byte_ofs - offset).bytes() as u8));
            }
        }
        self.ptrs.remove_range(ptr_range);

        // Overwrite bytewise provenance.
        for offset in range.start..range.end() {
            // The fragment index does not matter for wildcard provenance.
            bytes.insert(offset, (wildcard, 0));
        }
    }
}

/// A partial, owned list of provenance to transfer into another allocation.
///
/// Offsets are relative to the beginning of the copied range.
pub struct ProvenanceCopy<Prov> {
    ptrs: Box<[(Size, Prov)]>,
    bytes: Box<[(Size, (Prov, u8))]>,
}

impl<Prov: Provenance> ProvenanceMap<Prov> {
    pub fn prepare_copy(
        &self,
        range: AllocRange,
        cx: &impl HasDataLayout,
    ) -> AllocResult<ProvenanceCopy<Prov>> {
        let shift_offset = move |offset| offset - range.start;
        let ptr_size = cx.data_layout().pointer_size();

        // # Pointer-sized provenances
        // Get the provenances that are entirely within this range.
        // (Different from `range_get_ptrs` which asks if they overlap the range.)
        // Only makes sense if we are copying at least one pointer worth of bytes.
        let mut ptrs_box: Box<[_]> = Box::new([]);
        if range.size >= ptr_size {
            let adjusted_end = Size::from_bytes(range.end().bytes() - (ptr_size.bytes() - 1));
            let ptrs = self.ptrs.range(range.start..adjusted_end);
            ptrs_box = ptrs.iter().map(|&(offset, reloc)| (shift_offset(offset), reloc)).collect();
        };

        // # Byte-sized provenances
        // This includes the existing bytewise provenance in the range, and ptr provenance
        // that overlaps with the begin/end of the range.
        let mut bytes_box: Box<[_]> = Box::new([]);
        let begin_overlap = self.range_ptrs_get(alloc_range(range.start, Size::ZERO), cx).first();
        let end_overlap = self.range_ptrs_get(alloc_range(range.end(), Size::ZERO), cx).first();
        // We only need to go here if there is some overlap or some bytewise provenance.
        if begin_overlap.is_some() || end_overlap.is_some() || self.bytes.is_some() {
            let mut bytes: Vec<(Size, (Prov, u8))> = Vec::new();
            // First, if there is a part of a pointer at the start, add that.
            if let Some(entry) = begin_overlap {
                trace!("start overlapping entry: {entry:?}");
                // For really small copies, make sure we don't run off the end of the range.
                let entry_end = cmp::min(entry.0 + ptr_size, range.end());
                for offset in range.start..entry_end {
                    bytes.push((shift_offset(offset), (entry.1, (offset - entry.0).bytes() as u8)));
                }
            } else {
                trace!("no start overlapping entry");
            }

            // Then the main part, bytewise provenance from `self.bytes`.
            bytes.extend(
                self.range_bytes_get(range)
                    .iter()
                    .map(|&(offset, reloc)| (shift_offset(offset), reloc)),
            );

            // And finally possibly parts of a pointer at the end.
            if let Some(entry) = end_overlap {
                trace!("end overlapping entry: {entry:?}");
                // For really small copies, make sure we don't start before `range` does.
                let entry_start = cmp::max(entry.0, range.start);
                for offset in entry_start..range.end() {
                    if bytes.last().is_none_or(|bytes_entry| bytes_entry.0 < offset) {
                        // The last entry, if it exists, has a lower offset than us, so we
                        // can add it at the end and remain sorted.
                        bytes.push((
                            shift_offset(offset),
                            (entry.1, (offset - entry.0).bytes() as u8),
                        ));
                    } else {
                        // There already is an entry for this offset in there! This can happen when the
                        // start and end range checks actually end up hitting the same pointer, so we
                        // already added this in the "pointer at the start" part above.
                        assert!(entry.0 <= range.start);
                    }
                }
            } else {
                trace!("no end overlapping entry");
            }
            trace!("byte provenances: {bytes:?}");

            if !bytes.is_empty() && !Prov::OFFSET_IS_ADDR {
                // FIXME(#146291): We need to ensure that we don't mix different pointers with
                // the same provenance.
                return Err(AllocError::ReadPartialPointer(range.start));
            }

            // And again a buffer for the new list on the target side.
            bytes_box = bytes.into_boxed_slice();
        }

        Ok(ProvenanceCopy { ptrs: ptrs_box, bytes: bytes_box })
    }

    /// Applies a provenance copy.
    /// The affected range, as defined in the parameters to `prepare_copy` is expected
    /// to be clear of provenance.
    pub fn apply_copy(&mut self, copy: ProvenanceCopy<Prov>, range: AllocRange, repeat: u64) {
        let shift_offset = |idx: u64, offset: Size| offset + range.start + idx * range.size;
        if !copy.ptrs.is_empty() {
            // We want to call `insert_presorted` only once so that, if possible, the entries
            // after the range we insert are moved back only once.
            let chunk_len = copy.ptrs.len() as u64;
            self.ptrs.insert_presorted((0..chunk_len * repeat).map(|i| {
                let chunk = i / chunk_len;
                let (offset, reloc) = copy.ptrs[(i % chunk_len) as usize];
                (shift_offset(chunk, offset), reloc)
            }));
        }
        if !copy.bytes.is_empty() {
            let chunk_len = copy.bytes.len() as u64;
            self.bytes.get_or_insert_with(Box::default).insert_presorted(
                (0..chunk_len * repeat).map(|i| {
                    let chunk = i / chunk_len;
                    let (offset, reloc) = copy.bytes[(i % chunk_len) as usize];
                    (shift_offset(chunk, offset), reloc)
                }),
            );
        }
    }
}
