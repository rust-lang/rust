use byteorder::{ByteOrder, NativeEndian, ReadBytesExt, WriteBytesExt};
use std::collections::Bound::{Included, Excluded};
use std::collections::{btree_map, BTreeMap, HashMap, HashSet, VecDeque};
use std::{fmt, iter, mem, ptr};

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

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub struct AllocId(u64);

impl fmt::Display for AllocId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug)]
pub struct Allocation {
    pub bytes: Vec<u8>,
    pub relocations: BTreeMap<usize, AllocId>,
    pub undef_mask: UndefMask,
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
    alloc_map: HashMap<AllocId, Allocation>,
    next_id: AllocId,
    pub pointer_size: usize,
}

impl Memory {
    pub fn new() -> Self {
        Memory {
            alloc_map: HashMap::new(),
            next_id: AllocId(0),

            // FIXME(tsion): This should work for both 4 and 8, but it currently breaks some things
            // when set to 4.
            pointer_size: 8,
        }
    }

    pub fn allocate(&mut self, size: usize) -> Pointer {
        let alloc = Allocation {
            bytes: vec![0; size],
            relocations: BTreeMap::new(),
            undef_mask: UndefMask::new(size),
        };
        let id = self.next_id;
        self.next_id.0 += 1;
        self.alloc_map.insert(id, alloc);
        Pointer {
            alloc_id: id,
            offset: 0,
        }
    }

    // TODO(tsion): Track which allocations were returned from __rust_allocate and report an error
    // when reallocating/deallocating any others.
    pub fn reallocate(&mut self, ptr: Pointer, new_size: usize) -> EvalResult<()> {
        if ptr.offset != 0 {
            // TODO(tsion): Report error about non-__rust_allocate'd pointer.
            panic!()
        }

        let alloc = try!(self.get_mut(ptr.alloc_id));
        let size = alloc.bytes.len();
        if new_size > size {
            let amount = new_size - size;
            alloc.bytes.extend(iter::repeat(0).take(amount));
            alloc.undef_mask.grow(amount, false);
        } else if size > new_size {
            unimplemented!()
            // alloc.bytes.truncate(new_size);
            // alloc.undef_mask.len = new_size;
            // TODO: potentially remove relocations
        }

        Ok(())
    }

    // TODO(tsion): See comment on `reallocate`.
    pub fn deallocate(&mut self, ptr: Pointer) -> EvalResult<()> {
        if ptr.offset != 0 {
            // TODO(tsion): Report error about non-__rust_allocate'd pointer.
            panic!()
        }

        if self.alloc_map.remove(&ptr.alloc_id).is_none() {
            // TODO(tsion): Report error about erroneous free. This is blocked on properly tracking
            // already-dropped state since this if-statement is entered even in safe code without
            // it.
        }

        Ok(())
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Allocation accessors
    ////////////////////////////////////////////////////////////////////////////////

    pub fn get(&self, id: AllocId) -> EvalResult<&Allocation> {
        self.alloc_map.get(&id).ok_or(EvalError::DanglingPointerDeref)
    }

    pub fn get_mut(&mut self, id: AllocId) -> EvalResult<&mut Allocation> {
        self.alloc_map.get_mut(&id).ok_or(EvalError::DanglingPointerDeref)
    }

    /// Print an allocation and all allocations it points to, recursively.
    pub fn dump(&self, id: AllocId) {
        let mut allocs_seen = HashSet::new();
        let mut allocs_to_print = VecDeque::new();
        allocs_to_print.push_back(id);

        while let Some(id) = allocs_to_print.pop_front() {
            allocs_seen.insert(id);
            let prefix = format!("Alloc {:<5} ", format!("{}:", id));
            print!("{}", prefix);
            let mut relocations = vec![];

            let alloc = match self.alloc_map.get(&id) {
                Some(a) => a,
                None => {
                    println!("(deallocated)");
                    continue;
                }
            };

            for i in 0..alloc.bytes.len() {
                if let Some(&target_id) = alloc.relocations.get(&i) {
                    if !allocs_seen.contains(&target_id) {
                        allocs_to_print.push_back(target_id);
                    }
                    relocations.push((i, target_id));
                }
                if alloc.undef_mask.is_range_defined(i, i + 1) {
                    print!("{:02x} ", alloc.bytes[i]);
                } else {
                    print!("__ ");
                }
            }
            println!("({} bytes)", alloc.bytes.len());

            if !relocations.is_empty() {
                print!("{:1$}", "", prefix.len()); // Print spaces.
                let mut pos = 0;
                let relocation_width = (self.pointer_size - 1) * 3;
                for (i, target_id) in relocations {
                    print!("{:1$}", "", (i - pos) * 3);
                    print!("└{0:─^1$}┘ ", format!("({})", target_id), relocation_width);
                    pos = i + self.pointer_size;
                }
                println!("");
            }
        }
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

        try!(self.copy_undef_mask(src, dest, size));
        try!(self.copy_relocations(src, dest, size));

        Ok(())
    }

    pub fn read_bytes(&self, ptr: Pointer, size: usize) -> EvalResult<&[u8]> {
        self.get_bytes(ptr, size)
    }

    pub fn write_bytes(&mut self, ptr: Pointer, src: &[u8]) -> EvalResult<()> {
        let bytes = try!(self.get_bytes_mut(ptr, src.len()));
        bytes.clone_from_slice(src);
        Ok(())
    }

    pub fn write_repeat(&mut self, ptr: Pointer, val: u8, count: usize) -> EvalResult<()> {
        let bytes = try!(self.get_bytes_mut(ptr, count));
        for b in bytes { *b = val; }
        Ok(())
    }

    pub fn drop_fill(&mut self, ptr: Pointer, size: usize) -> EvalResult<()> {
        self.write_repeat(ptr, mem::POST_DROP_U8, size)
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
        if first < start { alloc.undef_mask.set_range(first, start, false); }
        if last > end { alloc.undef_mask.set_range(end, last, false); }

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

    // FIXME(tsino): This is a very naive, slow version.
    fn copy_undef_mask(&mut self, src: Pointer, dest: Pointer, size: usize) -> EvalResult<()> {
        // The bits have to be saved locally before writing to dest in case src and dest overlap.
        let mut v = Vec::with_capacity(size);
        for i in 0..size {
            let defined = try!(self.get(src.alloc_id)).undef_mask.get(src.offset + i);
            v.push(defined);
        }
        for (i, defined) in v.into_iter().enumerate() {
            try!(self.get_mut(dest.alloc_id)).undef_mask.set(dest.offset + i, defined);
        }
        Ok(())
    }

    fn check_defined(&self, ptr: Pointer, size: usize) -> EvalResult<()> {
        let alloc = try!(self.get(ptr.alloc_id));
        if !alloc.undef_mask.is_range_defined(ptr.offset, ptr.offset + size) {
            return Err(EvalError::ReadUndefBytes);
        }
        Ok(())
    }

    pub fn mark_definedness(&mut self, ptr: Pointer, size: usize, new_state: bool)
        -> EvalResult<()>
    {
        let mut alloc = try!(self.get_mut(ptr.alloc_id));
        alloc.undef_mask.set_range(ptr.offset, ptr.offset + size, new_state);
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////
// Undefined byte tracking
////////////////////////////////////////////////////////////////////////////////

type Block = u64;
const BLOCK_SIZE: usize = 64;

#[derive(Clone, Debug)]
pub struct UndefMask {
    blocks: Vec<Block>,
    len: usize,
}

impl UndefMask {
    fn new(size: usize) -> Self {
        let mut m = UndefMask {
            blocks: vec![],
            len: 0,
        };
        m.grow(size, false);
        m
    }

    /// Check whether the range `start..end` (end-exclusive) is entirely defined.
    fn is_range_defined(&self, start: usize, end: usize) -> bool {
        if end > self.len { return false; }
        for i in start..end {
            if !self.get(i) { return false; }
        }
        true
    }

    fn set_range(&mut self, start: usize, end: usize, new_state: bool) {
        let len = self.len;
        if end > len { self.grow(end - len, new_state); }
        self.set_range_inbounds(start, end, new_state);
    }

    fn set_range_inbounds(&mut self, start: usize, end: usize, new_state: bool) {
        for i in start..end { self.set(i, new_state); }
    }

    fn get(&self, i: usize) -> bool {
        let (block, bit) = bit_index(i);
        (self.blocks[block] & 1 << bit) != 0
    }

    fn set(&mut self, i: usize, new_state: bool) {
        let (block, bit) = bit_index(i);
        if new_state {
            self.blocks[block] |= 1 << bit;
        } else {
            self.blocks[block] &= !(1 << bit);
        }
    }

    fn grow(&mut self, amount: usize, new_state: bool) {
        let unused_trailing_bits = self.blocks.len() * BLOCK_SIZE - self.len;
        if amount > unused_trailing_bits {
            let additional_blocks = amount / BLOCK_SIZE + 1;
            self.blocks.extend(iter::repeat(0).take(additional_blocks));
        }
        let start = self.len;
        self.len += amount;
        self.set_range_inbounds(start, start + amount, new_state);
    }
}

// fn uniform_block(state: bool) -> Block {
//     if state { !0 } else { 0 }
// }

fn bit_index(bits: usize) -> (usize, usize) {
    (bits / BLOCK_SIZE, bits % BLOCK_SIZE)
}
