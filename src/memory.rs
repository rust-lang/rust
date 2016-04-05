use byteorder::{ByteOrder, NativeEndian, ReadBytesExt, WriteBytesExt};
use std::collections::{btree_map, BTreeMap, HashMap};
use std::collections::Bound::{Included, Excluded};
use std::mem;
use std::ptr;

use error::{EvalError, EvalResult};
use primval::PrimVal;

////////////////////////////////////////////////////////////////////////////////
// Value representations
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Repr {
    /// Representation for a non-aggregate type such as a boolean, integer, character or pointer.
    Primitive {
        size: usize
    },

    /// The representation for aggregate types including structs, enums, and tuples.
    Aggregate {
        /// The size of the discriminant (an integer). Should be between 0 and 8. Always 0 for
        /// structs and tuples.
        discr_size: usize,

        /// The size of the entire aggregate, including the discriminant.
        size: usize,

        /// The representations of the contents of each variant.
        variants: Vec<Vec<FieldRepr>>,
    },

    Array {
        elem_size: usize,

        /// Number of elements.
        length: usize,
    },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FieldRepr {
    pub offset: usize,
    pub size: usize,
}

impl Repr {
    pub fn size(&self) -> usize {
        match *self {
            Repr::Primitive { size } => size,
            Repr::Aggregate { size, .. } => size,
            Repr::Array { elem_size, length } => elem_size * length,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Allocations and pointers
////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AllocId(u64);

#[derive(Debug)]
pub struct Allocation {
    pub bytes: Box<[u8]>,
    pub relocations: BTreeMap<usize, AllocId>,
    pub undef_mask: Option<Vec<usize>>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Pointer {
    pub alloc_id: AllocId,
    pub offset: usize,
}

impl Pointer {
    pub fn offset(self, i: isize) -> Self {
        Pointer { offset: (self.offset as isize + i) as usize, ..self }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Top-level interpreter memory
////////////////////////////////////////////////////////////////////////////////

pub struct Memory {
    alloc_map: HashMap<u64, Allocation>,
    next_id: u64,
    pub pointer_size: usize,
}

impl Memory {
    pub fn new() -> Self {
        Memory {
            alloc_map: HashMap::new(),
            next_id: 0,

            // TODO(tsion): Should this be host's or target's usize?
            pointer_size: mem::size_of::<usize>(),
        }
    }

    pub fn allocate(&mut self, size: usize) -> Pointer {
        let id = AllocId(self.next_id);
        let alloc = Allocation {
            bytes: vec![0; size].into_boxed_slice(),
            relocations: BTreeMap::new(),
            undef_mask: None,
        };
        self.alloc_map.insert(self.next_id, alloc);
        self.next_id += 1;
        Pointer {
            alloc_id: id,
            offset: 0,
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Allocation accessors
    ////////////////////////////////////////////////////////////////////////////////

    pub fn get(&self, id: AllocId) -> EvalResult<&Allocation> {
        self.alloc_map.get(&id.0).ok_or(EvalError::DanglingPointerDeref)
    }

    pub fn get_mut(&mut self, id: AllocId) -> EvalResult<&mut Allocation> {
        self.alloc_map.get_mut(&id.0).ok_or(EvalError::DanglingPointerDeref)
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Byte accessors
    ////////////////////////////////////////////////////////////////////////////////

    fn get_bytes_unchecked(&self, ptr: Pointer, size: usize) -> EvalResult<&[u8]> {
        let alloc = try!(self.get(ptr.alloc_id));
        if ptr.offset + size > alloc.bytes.len() {
            return Err(EvalError::PointerOutOfBounds);
        }
        Ok(&alloc.bytes[ptr.offset..ptr.offset + size])
    }

    fn get_bytes_unchecked_mut(&mut self, ptr: Pointer, size: usize) -> EvalResult<&mut [u8]> {
        let alloc = try!(self.get_mut(ptr.alloc_id));
        if ptr.offset + size > alloc.bytes.len() {
            return Err(EvalError::PointerOutOfBounds);
        }
        Ok(&mut alloc.bytes[ptr.offset..ptr.offset + size])
    }

    fn get_bytes(&self, ptr: Pointer, size: usize) -> EvalResult<&[u8]> {
        if try!(self.relocations(ptr, size)).count() != 0 {
            return Err(EvalError::ReadPointerAsBytes);
        }
        try!(self.check_defined(ptr, size));
        self.get_bytes_unchecked(ptr, size)
    }

    fn get_bytes_mut(&mut self, ptr: Pointer, size: usize) -> EvalResult<&mut [u8]> {
        try!(self.clear_relocations(ptr, size));
        try!(self.mark_definedness(ptr, size, true));
        self.get_bytes_unchecked_mut(ptr, size)
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Reading and writing
    ////////////////////////////////////////////////////////////////////////////////

    pub fn copy(&mut self, src: Pointer, dest: Pointer, size: usize) -> EvalResult<()> {
        try!(self.check_relocation_edges(src, size));

        let src_bytes = try!(self.get_bytes_unchecked_mut(src, size)).as_mut_ptr();
        let dest_bytes = try!(self.get_bytes_mut(dest, size)).as_mut_ptr();

        // SAFE: The above indexing would have panicked if there weren't at least `size` bytes
        // behind `src` and `dest`. Also, we use the overlapping-safe `ptr::copy` if `src` and
        // `dest` could possibly overlap.
        unsafe {
            if src.alloc_id == dest.alloc_id {
                ptr::copy(src_bytes, dest_bytes, size);
            } else {
                ptr::copy_nonoverlapping(src_bytes, dest_bytes, size);
            }
        }

        // TODO(tsion): Copy undef ranges from src to dest.
        self.copy_relocations(src, dest, size)
    }

    pub fn write_bytes(&mut self, ptr: Pointer, src: &[u8]) -> EvalResult<()> {
        self.get_bytes_mut(ptr, src.len()).map(|dest| dest.clone_from_slice(src))
    }

    pub fn read_ptr(&self, ptr: Pointer) -> EvalResult<Pointer> {
        let size = self.pointer_size;
        try!(self.check_defined(ptr, size));
        let offset = try!(self.get_bytes_unchecked(ptr, size))
            .read_uint::<NativeEndian>(size).unwrap() as usize;
        let alloc = try!(self.get(ptr.alloc_id));
        match alloc.relocations.get(&ptr.offset) {
            Some(&alloc_id) => Ok(Pointer { alloc_id: alloc_id, offset: offset }),
            None => Err(EvalError::ReadBytesAsPointer),
        }
    }

    pub fn write_ptr(&mut self, dest: Pointer, ptr: Pointer) -> EvalResult<()> {
        {
            let size = self.pointer_size;
            let mut bytes = try!(self.get_bytes_mut(dest, size));
            bytes.write_uint::<NativeEndian>(ptr.offset as u64, size).unwrap();
        }
        try!(self.get_mut(dest.alloc_id)).relocations.insert(dest.offset, ptr.alloc_id);
        Ok(())
    }

    pub fn write_primval(&mut self, ptr: Pointer, val: PrimVal) -> EvalResult<()> {
        let pointer_size = self.pointer_size;
        match val {
            PrimVal::Bool(b) => self.write_bool(ptr, b),
            PrimVal::I8(n)   => self.write_int(ptr, n as i64, 1),
            PrimVal::I16(n)  => self.write_int(ptr, n as i64, 2),
            PrimVal::I32(n)  => self.write_int(ptr, n as i64, 4),
            PrimVal::I64(n)  => self.write_int(ptr, n as i64, 8),
            PrimVal::U8(n)   => self.write_uint(ptr, n as u64, 1),
            PrimVal::U16(n)  => self.write_uint(ptr, n as u64, 2),
            PrimVal::U32(n)  => self.write_uint(ptr, n as u64, 4),
            PrimVal::U64(n)  => self.write_uint(ptr, n as u64, 8),
            PrimVal::IntegerPtr(n) => self.write_uint(ptr, n as u64, pointer_size),
            PrimVal::AbstractPtr(_p) => unimplemented!(),
        }
    }

    pub fn read_bool(&self, ptr: Pointer) -> EvalResult<bool> {
        let bytes = try!(self.get_bytes(ptr, 1));
        match bytes[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(EvalError::InvalidBool),
        }
    }

    pub fn write_bool(&mut self, ptr: Pointer, b: bool) -> EvalResult<()> {
        self.get_bytes_mut(ptr, 1).map(|bytes| bytes[0] = b as u8)
    }

    pub fn read_int(&self, ptr: Pointer, size: usize) -> EvalResult<i64> {
        self.get_bytes(ptr, size).map(|mut b| b.read_int::<NativeEndian>(size).unwrap())
    }

    pub fn write_int(&mut self, ptr: Pointer, n: i64, size: usize) -> EvalResult<()> {
        self.get_bytes_mut(ptr, size).map(|mut b| b.write_int::<NativeEndian>(n, size).unwrap())
    }

    pub fn read_uint(&self, ptr: Pointer, size: usize) -> EvalResult<u64> {
        self.get_bytes(ptr, size).map(|mut b| b.read_uint::<NativeEndian>(size).unwrap())
    }

    pub fn write_uint(&mut self, ptr: Pointer, n: u64, size: usize) -> EvalResult<()> {
        self.get_bytes_mut(ptr, size).map(|mut b| b.write_uint::<NativeEndian>(n, size).unwrap())
    }

    pub fn read_isize(&self, ptr: Pointer) -> EvalResult<i64> {
        self.read_int(ptr, self.pointer_size)
    }

    pub fn write_isize(&mut self, ptr: Pointer, n: i64) -> EvalResult<()> {
        let size = self.pointer_size;
        self.write_int(ptr, n, size)
    }

    pub fn read_usize(&self, ptr: Pointer) -> EvalResult<u64> {
        self.read_uint(ptr, self.pointer_size)
    }

    pub fn write_usize(&mut self, ptr: Pointer, n: u64) -> EvalResult<()> {
        let size = self.pointer_size;
        self.write_uint(ptr, n, size)
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Relocations
    ////////////////////////////////////////////////////////////////////////////////

    fn relocations(&self, ptr: Pointer, size: usize)
        -> EvalResult<btree_map::Range<usize, AllocId>>
    {
        let start = ptr.offset.saturating_sub(self.pointer_size - 1);
        let end = start + size;
        Ok(try!(self.get(ptr.alloc_id)).relocations.range(Included(&start), Excluded(&end)))
    }

    fn clear_relocations(&mut self, ptr: Pointer, size: usize) -> EvalResult<()> {
        // Find all relocations overlapping the given range.
        let keys: Vec<_> = try!(self.relocations(ptr, size)).map(|(&k, _)| k).collect();
        if keys.len() == 0 { return Ok(()); }

        // Find the start and end of the given range and its outermost relocations.
        let start = ptr.offset;
        let end = start + size;
        let first = *keys.first().unwrap();
        let last = *keys.last().unwrap() + self.pointer_size;

        let alloc = try!(self.get_mut(ptr.alloc_id));

        // Mark parts of the outermost relocations as undefined if they partially fall outside the
        // given range.
        if first < start { alloc.mark_definedness(first, start, false); }
        if last > end { alloc.mark_definedness(end, last, false); }

        // Forget all the relocations.
        for k in keys { alloc.relocations.remove(&k); }

        Ok(())
    }

    fn check_relocation_edges(&self, ptr: Pointer, size: usize) -> EvalResult<()> {
        let overlapping_start = try!(self.relocations(ptr, 0)).count();
        let overlapping_end = try!(self.relocations(ptr.offset(size as isize), 0)).count();
        if overlapping_start + overlapping_end != 0 {
            return Err(EvalError::ReadPointerAsBytes);
        }
        Ok(())
    }

    fn copy_relocations(&mut self, src: Pointer, dest: Pointer, size: usize) -> EvalResult<()> {
        let relocations: Vec<_> = try!(self.relocations(src, size))
            .map(|(&offset, &alloc_id)| {
                // Update relocation offsets for the new positions in the destination allocation.
                (offset + dest.offset - src.offset, alloc_id)
            })
            .collect();
        try!(self.get_mut(dest.alloc_id)).relocations.extend(relocations);
        Ok(())
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Undefined bytes
    ////////////////////////////////////////////////////////////////////////////////

    fn check_defined(&self, ptr: Pointer, size: usize) -> EvalResult<()> {
        let alloc = try!(self.get(ptr.alloc_id));
        if !alloc.is_range_defined(ptr.offset, ptr.offset + size) {
            return Err(EvalError::ReadUndefBytes);
        }
        Ok(())
    }

    pub fn mark_definedness(&mut self, ptr: Pointer, size: usize, new_state: bool)
        -> EvalResult<()>
    {
        let mut alloc = try!(self.get_mut(ptr.alloc_id));
        alloc.mark_definedness(ptr.offset, ptr.offset + size, new_state);
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////
// Undefined byte tracking
////////////////////////////////////////////////////////////////////////////////

impl Allocation {
    /// Check whether the range `start..end` (end-exclusive) in this allocation is entirely
    /// defined.
    fn is_range_defined(&self, start: usize, end: usize) -> bool {
        debug_assert!(start <= end);
        debug_assert!(end <= self.bytes.len());

        // An empty range is always fully defined.
        if start == end {
            return true;
        }

        match self.undef_mask {
            Some(ref undef_mask) => {
                // If `start` lands directly on a boundary, it belongs to the range after the
                // boundary, hence the increment in the `Ok` arm.
                let i = match undef_mask.binary_search(&start) { Ok(j) => j + 1, Err(j) => j };

                // The range is fully defined if and only if both:
                //   1. The start value falls into a defined range (with even parity).
                //   2. The end value is in the same range as the start value.
                i % 2 == 0 && undef_mask.get(i).map(|&x| end <= x).unwrap_or(true)
            }
            None => false,
        }
    }

    /// Mark the range `start..end` (end-exclusive) as defined or undefined, depending on
    /// `new_state`.
    fn mark_definedness(&mut self, start: usize, end: usize, new_state: bool) {
        debug_assert!(start <= end);
        debug_assert!(end <= self.bytes.len());

        // There is no need to track undef masks for zero-sized allocations.
        let len = self.bytes.len();
        if len == 0 {
            return;
        }

        // Returns whether the new state matches the state of a given undef mask index. The way
        // undef masks are represented, boundaries at even indices are undefined and those at odd
        // indices are defined.
        let index_matches_new_state = |i| i % 2 == new_state as usize;

        // Lookup the undef mask index where the given endpoint `i` is or should be inserted.
        let lookup_endpoint = |undef_mask: &[usize], i: usize| -> (usize, bool) {
            let (index, should_insert);
            match undef_mask.binary_search(&i) {
                // Region endpoint is on an undef mask boundary.
                Ok(j) => {
                    // This endpoint's index must be incremented if the boundary's state matches
                    // the region's new state so that the boundary is:
                    //   1. Excluded from deletion when handling the inclusive left-hand endpoint.
                    //   2. Included for deletion when handling the exclusive right-hand endpoint.
                    index = j + index_matches_new_state(j) as usize;

                    // Don't insert a new mask boundary; simply reuse or delete the matched one.
                    should_insert = false;
                }

                // Region endpoint is not on a mask boundary.
                Err(j) => {
                    // This is the index after the nearest mask boundary which has the same state.
                    index = j;

                    // Insert a new boundary if this endpoint's state doesn't match the state of
                    // this position.
                    should_insert = index_matches_new_state(j);
                }
            }
            (index, should_insert)
        };

        match self.undef_mask {
            // There is an existing undef mask, with arbitrary existing boundaries.
            Some(ref mut undef_mask) => {
                // Determine where the new range's endpoints fall within the current undef mask.
                let (start_index, insert_start) = lookup_endpoint(undef_mask, start);
                let (end_index, insert_end) = lookup_endpoint(undef_mask, end);

                // Delete all the undef mask boundaries overwritten by the new range.
                undef_mask.drain(start_index..end_index);

                // Insert any new boundaries deemed necessary with two exceptions:
                //   1. Never insert an endpoint equal to the allocation length; it's implicit.
                //   2. Never insert a start boundary equal to the end boundary.
                if insert_end && end != len {
                    undef_mask.insert(start_index, end);
                }
                if insert_start && start != end {
                    undef_mask.insert(start_index, start);
                }
            }

            // There is no existing undef mask. This is taken as meaning the entire allocation is
            // currently undefined. If the new state is false, meaning undefined, do nothing.
            None => if new_state {
                let mut mask = if start == 0 {
                    // 0..end is defined.
                    Vec::new()
                } else {
                    // 0..0 is defined, 0..start is undefined, start..end is defined.
                    vec![0, start]
                };

                // Don't insert the end boundary if it's equal to the allocation length; that
                // boundary is implicit.
                if end != len {
                    mask.push(end);
                }
                self.undef_mask = Some(mask);
            },
        }
    }
}

#[cfg(test)]
mod test {
    use memory::Allocation;
    use std::collections::BTreeMap;

    fn alloc_with_mask(len: usize, undef_mask: Option<Vec<usize>>) -> Allocation {
        Allocation {
            bytes: vec![0; len].into_boxed_slice(),
            relocations: BTreeMap::new(),
            undef_mask: undef_mask,
        }
    }

    #[test]
    fn large_undef_mask() {
        let mut alloc = alloc_with_mask(20, Some(vec![4, 8, 12, 16]));

        assert!(alloc.is_range_defined(0, 0));
        assert!(alloc.is_range_defined(0, 3));
        assert!(alloc.is_range_defined(0, 4));
        assert!(alloc.is_range_defined(1, 3));
        assert!(alloc.is_range_defined(1, 4));
        assert!(alloc.is_range_defined(4, 4));
        assert!(!alloc.is_range_defined(0, 5));
        assert!(!alloc.is_range_defined(1, 5));
        assert!(!alloc.is_range_defined(4, 5));
        assert!(!alloc.is_range_defined(4, 8));
        assert!(alloc.is_range_defined(8, 12));
        assert!(!alloc.is_range_defined(12, 16));
        assert!(alloc.is_range_defined(16, 20));
        assert!(!alloc.is_range_defined(15, 20));
        assert!(!alloc.is_range_defined(0, 20));

        alloc.mark_definedness(8, 11, false);
        assert_eq!(alloc.undef_mask, Some(vec![4, 11, 12, 16]));

        alloc.mark_definedness(8, 11, true);
        assert_eq!(alloc.undef_mask, Some(vec![4, 8, 12, 16]));

        alloc.mark_definedness(8, 12, false);
        assert_eq!(alloc.undef_mask, Some(vec![4, 16]));

        alloc.mark_definedness(8, 12, true);
        assert_eq!(alloc.undef_mask, Some(vec![4, 8, 12, 16]));

        alloc.mark_definedness(9, 11, true);
        assert_eq!(alloc.undef_mask, Some(vec![4, 8, 12, 16]));

        alloc.mark_definedness(9, 11, false);
        assert_eq!(alloc.undef_mask, Some(vec![4, 8, 9, 11, 12, 16]));

        alloc.mark_definedness(9, 10, true);
        assert_eq!(alloc.undef_mask, Some(vec![4, 8, 10, 11, 12, 16]));

        alloc.mark_definedness(8, 12, true);
        assert_eq!(alloc.undef_mask, Some(vec![4, 8, 12, 16]));
    }

    #[test]
    fn empty_undef_mask() {
        let mut alloc = alloc_with_mask(0, None);
        assert!(alloc.is_range_defined(0, 0));

        alloc.mark_definedness(0, 0, false);
        assert_eq!(alloc.undef_mask, None);
        assert!(alloc.is_range_defined(0, 0));

        alloc.mark_definedness(0, 0, true);
        assert_eq!(alloc.undef_mask, None);
        assert!(alloc.is_range_defined(0, 0));
    }

    #[test]
    fn small_undef_mask() {
        let mut alloc = alloc_with_mask(8, None);

        alloc.mark_definedness(0, 4, false);
        assert_eq!(alloc.undef_mask, None);

        alloc.mark_definedness(0, 4, true);
        assert_eq!(alloc.undef_mask, Some(vec![4]));

        alloc.mark_definedness(4, 8, false);
        assert_eq!(alloc.undef_mask, Some(vec![4]));

        alloc.mark_definedness(4, 8, true);
        assert_eq!(alloc.undef_mask, Some(vec![]));

        alloc.mark_definedness(0, 8, true);
        assert_eq!(alloc.undef_mask, Some(vec![]));

        alloc.mark_definedness(0, 8, false);
        assert_eq!(alloc.undef_mask, Some(vec![0]));

        alloc.mark_definedness(0, 8, true);
        assert_eq!(alloc.undef_mask, Some(vec![]));

        alloc.mark_definedness(4, 8, false);
        assert_eq!(alloc.undef_mask, Some(vec![4]));

        alloc.mark_definedness(0, 8, false);
        assert_eq!(alloc.undef_mask, Some(vec![0]));

        alloc.mark_definedness(2, 5, true);
        assert_eq!(alloc.undef_mask, Some(vec![0, 2, 5]));

        alloc.mark_definedness(4, 6, false);
        assert_eq!(alloc.undef_mask, Some(vec![0, 2, 4]));

        alloc.mark_definedness(0, 3, true);
        assert_eq!(alloc.undef_mask, Some(vec![4]));

        alloc.mark_definedness(2, 6, true);
        assert_eq!(alloc.undef_mask, Some(vec![6]));

        alloc.mark_definedness(3, 7, false);
        assert_eq!(alloc.undef_mask, Some(vec![3]));
    }
}
