//! Store the provenance for each byte in the range, with a more efficient
//! representation for the common case where PTR_SIZE consecutive bytes have the same provenance.

use std::cmp;
use std::ops::{Range, RangeBounds};

use rustc_abi::{HasDataLayout, Size};
use rustc_data_structures::sorted_map::SortedMap;
use rustc_macros::HashStable;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use tracing::trace;

use super::{AllocRange, CtfeProvenance, Provenance, alloc_range};
use crate::mir::interpret::{AllocError, AllocResult};

/// A pointer fragment represents one byte of a pointer.
/// If the bytes are re-assembled in their original order, the pointer can be used again.
/// Wildcard provenance is allowed to have index 0 everywhere.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[derive(HashStable)]
pub struct PointerFrag<Prov> {
    /// The position of this fragment inside the pointer (in `0..8`).
    pub idx: u8,
    /// The provenance of the pointer this is a fragment of.
    pub prov: Prov,
    /// The raw bytes of the pointer this is a fragment of.
    /// This is taken as a direct subslice of the raw allocation data, so we don't have to worry
    /// about endianness. If the pointer size is less than 8, only the first N bytes of this are
    /// ever non-zero.
    pub bytes: [u8; 8],
}

/// Stores the provenance information of pointers stored in memory.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[derive(HashStable)]
pub struct ProvenanceMap<Prov = CtfeProvenance> {
    /// `Provenance` in this map applies from the given offset for an entire pointer-size worth of
    /// bytes. Two entries in this map are always at least a pointer size apart.
    ptrs: SortedMap<Size, Prov>,
    /// This stores byte-sized provenance fragments.
    bytes: Option<Box<SortedMap<Size, PointerFrag<Prov>>>>,
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
    /// by offset and contain no duplicates.
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
    fn range_ptrs_get(&self, range: AllocRange, cx: &impl HasDataLayout) -> &[(Size, Prov)] {
        self.ptrs.range(Self::adjusted_range_ptrs(range, cx))
    }

    /// `pm.range_ptrs_is_empty(r, cx)` == `pm.range_ptrs_get(r, cx).is_empty()`, but is faster.
    fn range_ptrs_is_empty(&self, range: AllocRange, cx: &impl HasDataLayout) -> bool {
        self.ptrs.range_is_empty(Self::adjusted_range_ptrs(range, cx))
    }

    /// Check if there is ptr-sized provenance at the given index.
    /// Does not mean anything for bytewise provenance! But can be useful as an optimization.
    pub fn get_ptr(&self, offset: Size) -> Option<Prov> {
        self.ptrs.get(&offset).copied()
    }

    /// Returns all byte-wise provenance in the given range.
    fn range_bytes_get(&self, range: AllocRange) -> &[(Size, PointerFrag<Prov>)] {
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

    /// Get the provenance of a single byte. Must only be called if there is no
    /// pointer-sized provenance here.
    pub fn get_byte(&self, offset: Size, cx: &impl HasDataLayout) -> Option<&PointerFrag<Prov>> {
        debug_assert!(self.range_ptrs_is_empty(alloc_range(offset, Size::from_bytes(1)), cx));
        self.bytes.as_ref().and_then(|b| b.get(&offset))
    }

    /// Gets the provenances of all bytes (including from pointers) in a range.
    pub fn get_range(
        &self,
        range: AllocRange,
        cx: &impl HasDataLayout,
    ) -> impl Iterator<Item = (AllocRange, Prov)> {
        let ptr_size = cx.data_layout().pointer_size();
        let ptr_provs = self
            .range_ptrs_get(range, cx)
            .iter()
            .map(move |(offset, p)| (alloc_range(*offset, ptr_size), *p));
        let byte_provs = self
            .range_bytes_get(range)
            .iter()
            .map(move |(offset, frag)| (alloc_range(*offset, Size::from_bytes(1)), frag.prov));
        ptr_provs.chain(byte_provs)
    }

    /// Attempt to merge per-byte provenance back into ptr chunks, if the right fragments
    /// sit next to each other. Return `false` if that is not possible due to partial pointers.
    pub fn merge_bytes(&mut self, cx: &impl HasDataLayout) -> bool {
        let Some(bytes) = self.bytes.as_deref_mut() else {
            return true;
        };
        let ptr_size = cx.data_layout().pointer_size();
        while let Some((offset, first_frag)) = bytes.iter().next() {
            let offset = *offset;
            // Check if this fragment starts a pointer.
            let range = offset..offset + ptr_size;
            let frags = bytes.range(range.clone());
            if frags.len() != ptr_size.bytes_usize() {
                // We can't merge this one, no point in trying to merge the rest.
                return false;
            }
            for (idx, (_offset, frag)) in frags.iter().enumerate() {
                if !(frag.prov == first_frag.prov
                    && frag.bytes == first_frag.bytes
                    && frag.idx == idx as u8)
                {
                    return false;
                }
            }
            // Looks like a pointer! Move it over to the ptr provenance map.
            self.ptrs.insert(offset, first_frag.prov);
            bytes.remove_range(range);
        }
        // We managed to convert everything into whole pointers.
        self.bytes = None;
        true
    }

    /// Try to read a pointer from the given location, possibly by loading from many per-byte
    /// provenances.
    pub fn read_ptr(&self, offset: Size, cx: &impl HasDataLayout) -> AllocResult<Option<Prov>> {
        // If there is pointer-sized provenance exactly here, we can just return that.
        if let Some(prov) = self.get_ptr(offset) {
            return Ok(Some(prov));
        }
        // The other easy case is total absence of provenance, that also always works.
        let range = alloc_range(offset, cx.data_layout().pointer_size());
        let no_ptrs = self.range_ptrs_is_empty(range, cx);
        if no_ptrs && self.range_bytes_is_empty(range) {
            return Ok(None);
        }
        // If we get here, we have to check whether we can merge per-byte provenance.
        let prov = 'prov: {
            // If there is any ptr-sized provenance overlapping with this range,
            // this is definitely mixing multiple pointers and we can bail.
            if !no_ptrs {
                break 'prov None;
            }
            // Scan all fragments, and ensure their indices, provenance, and bytes match.
            // However, we have to ignore wildcard fragments for this (this is needed for Miri's
            // native-lib mode). Therefore, we will only know the expected provenance and bytes
            // once we find the first non-wildcard fragment.
            let mut expected = None;
            for idx in Size::ZERO..range.size {
                // Ensure there is provenance here.
                let Some(frag) = self.get_byte(offset + idx, cx) else {
                    break 'prov None;
                };
                // If this is wildcard provenance, ignore this fragment.
                if Some(frag.prov) == Prov::WILDCARD {
                    continue;
                }
                // For non-wildcard fragments, the index must match.
                if u64::from(frag.idx) != idx.bytes() {
                    break 'prov None;
                }
                // If there are expectations registered, check them.
                // If not, record this fragment as setting the expectations.
                match expected {
                    Some(expected) => {
                        if (frag.prov, frag.bytes) != expected {
                            break 'prov None;
                        }
                    }
                    None => {
                        expected = Some((frag.prov, frag.bytes));
                    }
                }
            }
            // The final provenance is the expected one we found along the way, or wildcard if
            // we didn't find any.
            Some(expected.map(|(prov, _addr)| prov).or_else(|| Prov::WILDCARD).unwrap())
        };
        if prov.is_none() && !Prov::OFFSET_IS_ADDR {
            // There are some bytes with provenance here but overall the provenance does not add up.
            // We need `OFFSET_IS_ADDR` to fall back to no-provenance here; without that option, we
            // must error.
            return Err(AllocError::ReadPartialPointer(offset));
        }
        Ok(prov)
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
        let bytes = self.bytes.iter().flat_map(|b| b.values().map(|frag| frag.prov));
        self.ptrs.values().copied().chain(bytes)
    }

    pub fn insert_ptr(&mut self, offset: Size, prov: Prov, cx: &impl HasDataLayout) {
        debug_assert!(self.range_empty(alloc_range(offset, cx.data_layout().pointer_size()), cx));
        self.ptrs.insert(offset, prov);
    }

    /// Returns an iterator that yields the fragments of this pointer whose absolute positions are
    /// inside `pos_range`.
    fn ptr_fragments(
        pos_range: impl RangeBounds<Size>,
        ptr_pos: Size,
        prov: Prov,
        data_bytes: &[u8],
        ptr_size: Size,
    ) -> impl Iterator<Item = (Size, PointerFrag<Prov>)> {
        if pos_range.is_empty() {
            return either::Left(std::iter::empty());
        }
        // Read ptr_size many bytes starting at ptr_pos.
        let mut bytes = [0u8; 8];
        (&mut bytes[..ptr_size.bytes_usize()])
            .copy_from_slice(&data_bytes[ptr_pos.bytes_usize()..][..ptr_size.bytes_usize()]);
        // Yield the fragments of this pointer.
        either::Right(
            (ptr_pos..ptr_pos + ptr_size).filter(move |pos| pos_range.contains(pos)).map(
                move |pos| (pos, PointerFrag { idx: (pos - ptr_pos).bytes() as u8, bytes, prov }),
            ),
        )
    }

    /// Removes all provenance inside the given range.
    #[allow(irrefutable_let_patterns)] // these actually make the code more clear
    pub fn clear(&mut self, range: AllocRange, data_bytes: &[u8], cx: &impl HasDataLayout) {
        if range.size == Size::ZERO {
            return;
        }

        let start = range.start;
        let end = range.end();
        // Clear the bytewise part -- this is easy.
        if let Some(bytes) = self.bytes.as_mut() {
            bytes.remove_range(start..end);
        }

        // Find all provenance overlapping the given range.
        let ptrs_range = Self::adjusted_range_ptrs(range, cx);
        if self.ptrs.range_is_empty(ptrs_range.clone()) {
            // No provenance in this range, we are done. This is the common case.
            return;
        }
        let pointer_size = cx.data_layout().pointer_size();

        // This redoes some of the work of `range_is_empty`, but this path is much
        // colder than the early return above, so it's worth it.
        let ptrs = self.ptrs.range(ptrs_range.clone());

        // We need to handle clearing the provenance from parts of a pointer.
        if let &(first, prov) = ptrs.first().unwrap()
            && first < start
        {
            // Insert the remaining part in the bytewise provenance.
            let bytes = self.bytes.get_or_insert_with(Box::default);
            for (pos, frag) in Self::ptr_fragments(..start, first, prov, data_bytes, pointer_size) {
                bytes.insert(pos, frag);
            }
        }
        if let &(last, prov) = ptrs.last().unwrap()
            && last + pointer_size > end
        {
            // Insert the remaining part in the bytewise provenance.
            let bytes = self.bytes.get_or_insert_with(Box::default);
            for (pos, frag) in Self::ptr_fragments(end.., last, prov, data_bytes, pointer_size) {
                bytes.insert(pos, frag);
            }
        }

        // Forget all the provenance.
        // Since provenance do not overlap, we know that removing until `last` (exclusive) is fine,
        // i.e., this will not remove any other provenance just after the ones we care about.
        self.ptrs.remove_range(ptrs_range);
    }

    /// Overwrites all provenance in the given range with wildcard provenance.
    /// Pointers partially overwritten will have their provenances preserved
    /// bytewise on their remaining bytes.
    ///
    /// Provided for usage in Miri and panics otherwise.
    pub fn write_wildcards(
        &mut self,
        cx: &impl HasDataLayout,
        data_bytes: &[u8],
        range: AllocRange,
    ) {
        let wildcard = Prov::WILDCARD.unwrap();

        // Clear existing provenance in this range.
        self.clear(range, data_bytes, cx);

        // Make everything in the range wildcards.
        let bytes = self.bytes.get_or_insert_with(Box::default);
        for offset in range.start..range.end() {
            // The fragment index and bytes do not matter for wildcard provenance.
            bytes.insert(
                offset,
                PointerFrag { prov: wildcard, idx: Default::default(), bytes: Default::default() },
            );
        }
    }
}

/// A partial, owned list of provenance to transfer into another allocation.
///
/// Offsets are relative to the beginning of the copied range.
pub struct ProvenanceCopy<Prov> {
    ptrs: Box<[(Size, Prov)]>,
    bytes: Box<[(Size, PointerFrag<Prov>)]>,
}

impl<Prov: Provenance> ProvenanceMap<Prov> {
    pub fn prepare_copy(
        &self,
        range: AllocRange,
        data_bytes: &[u8],
        cx: &impl HasDataLayout,
    ) -> ProvenanceCopy<Prov> {
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
            let mut bytes: Vec<(Size, PointerFrag<Prov>)> = Vec::new();
            // First, if there is a part of a pointer at the start, add that.
            if let Some(&(pos, prov)) = begin_overlap {
                // For really small copies, make sure we don't run off the end of the range.
                let end = cmp::min(pos + ptr_size, range.end());
                for (pos, frag) in
                    Self::ptr_fragments(range.start..end, pos, prov, data_bytes, ptr_size)
                {
                    bytes.push((shift_offset(pos), frag));
                }
            } else {
                trace!("no start overlapping entry");
            }

            // Then the main part, bytewise provenance from `self.bytes`.
            bytes.extend(
                self.range_bytes_get(range)
                    .iter()
                    .map(|(offset, frag)| (shift_offset(*offset), frag.clone())),
            );

            // And finally possibly parts of a pointer at the end.
            // We only have to go here if this is actually different than the begin_overlap.
            if let Some(&(pos, prov)) = end_overlap
                && begin_overlap.is_none_or(|(begin, _)| *begin != pos)
            {
                // If this was a really small copy, we'd have handled this in begin_overlap.
                assert!(pos >= range.start);
                for (pos, frag) in
                    Self::ptr_fragments(pos..range.end(), pos, prov, data_bytes, ptr_size)
                {
                    let pos = shift_offset(pos);
                    // The last entry, if it exists, has a lower offset than us, so we
                    // can add it at the end and remain sorted.
                    debug_assert!(bytes.last().is_none_or(|bytes_entry| bytes_entry.0 < pos));
                    bytes.push((pos, frag));
                }
            } else {
                trace!("no end overlapping entry");
            }
            trace!("byte provenances: {bytes:?}");

            // And again a buffer for the new list on the target side.
            bytes_box = bytes.into_boxed_slice();
        }

        ProvenanceCopy { ptrs: ptrs_box, bytes: bytes_box }
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
                let (offset, prov) = copy.ptrs[(i % chunk_len) as usize];
                (shift_offset(chunk, offset), prov)
            }));
        }
        if !copy.bytes.is_empty() {
            let chunk_len = copy.bytes.len() as u64;
            self.bytes.get_or_insert_with(Box::default).insert_presorted(
                (0..chunk_len * repeat).map(|i| {
                    let chunk = i / chunk_len;
                    let (offset, frag) = &copy.bytes[(i % chunk_len) as usize];
                    (shift_offset(chunk, *offset), frag.clone())
                }),
            );
        }
    }
}
