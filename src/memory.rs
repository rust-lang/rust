use byteorder::{ByteOrder, NativeEndian, ReadBytesExt, WriteBytesExt};
use std::collections::{btree_map, BTreeMap, HashMap};
use std::collections::Bound::{Included, Excluded};
use std::mem;
use std::ptr;

use error::{EvalError, EvalResult};
use primval::PrimVal;

pub struct Memory {
    alloc_map: HashMap<u64, Allocation>,
    next_id: u64,
    pub pointer_size: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AllocId(u64);

#[derive(Debug)]
pub struct Allocation {
    pub bytes: Box<[u8]>,
    pub relocations: BTreeMap<usize, AllocId>,
    // TODO(tsion): undef mask
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Pointer {
    pub alloc_id: AllocId,
    pub offset: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FieldRepr {
    pub offset: usize,
    pub size: usize,
}

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
        };
        self.alloc_map.insert(self.next_id, alloc);
        self.next_id += 1;
        Pointer {
            alloc_id: id,
            offset: 0,
        }
    }

    pub fn get(&self, id: AllocId) -> EvalResult<&Allocation> {
        self.alloc_map.get(&id.0).ok_or(EvalError::DanglingPointerDeref)
    }

    pub fn get_mut(&mut self, id: AllocId) -> EvalResult<&mut Allocation> {
        self.alloc_map.get_mut(&id.0).ok_or(EvalError::DanglingPointerDeref)
    }

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
        // TODO(tsion): Track and check for undef bytes.
        self.get_bytes_unchecked(ptr, size)
    }

    fn get_bytes_mut(&mut self, ptr: Pointer, size: usize) -> EvalResult<&mut [u8]> {
        try!(self.clear_relocations(ptr, size));
        self.get_bytes_unchecked_mut(ptr, size)
    }

    fn relocations(&self, ptr: Pointer, size: usize)
        -> EvalResult<btree_map::Range<usize, AllocId>>
    {
        let start = ptr.offset.saturating_sub(self.pointer_size - 1);
        let end = start + size;
        Ok(try!(self.get(ptr.alloc_id)).relocations.range(Included(&start), Excluded(&end)))
    }

    fn clear_relocations(&mut self, ptr: Pointer, size: usize) -> EvalResult<()> {
        let keys: Vec<_> = try!(self.relocations(ptr, size)).map(|(&k, _)| k).collect();
        let alloc = try!(self.get_mut(ptr.alloc_id));
        for k in keys {
            alloc.relocations.remove(&k);
        }
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

    pub fn copy(&mut self, src: Pointer, dest: Pointer, size: usize) -> EvalResult<()> {
        // TODO(tsion): Track and check for undef bytes.
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

        self.copy_relocations(src, dest, size)
    }

    pub fn write_bytes(&mut self, ptr: Pointer, src: &[u8]) -> EvalResult<()> {
        self.get_bytes_mut(ptr, src.len()).map(|dest| dest.clone_from_slice(src))
    }

    pub fn read_ptr(&self, ptr: Pointer) -> EvalResult<Pointer> {
        let size = self.pointer_size;
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
}

impl Pointer {
    pub fn offset(self, i: isize) -> Self {
        Pointer { offset: (self.offset as isize + i) as usize, ..self }
    }
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
