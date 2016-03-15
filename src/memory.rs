use byteorder::{self, ByteOrder};
use std::collections::{BTreeMap, HashMap};
use std::collections::Bound::{Included, Excluded};
use std::mem;
use std::ptr;

use error::{EvalError, EvalResult};
use primval::PrimVal;

const POINTER_SIZE: usize = 8;

pub struct Memory {
    next_id: u64,
    alloc_map: HashMap<u64, Allocation>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AllocId(u64);

#[derive(Debug)]
pub struct Allocation {
    pub bytes: Vec<u8>,
    pub relocations: BTreeMap<usize, AllocId>,
    // TODO(tsion): undef mask
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Pointer {
    pub alloc_id: AllocId,
    pub offset: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldRepr {
    pub offset: usize,
    pub repr: Repr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Repr {
    Bool,
    I8, I16, I32, I64,
    U8, U16, U32, U64,

    /// The representation for product types including tuples, structs, and the contents of enum
    /// variants.
    Product {
        /// Size in bytes.
        size: usize,
        fields: Vec<FieldRepr>,
    },

    /// The representation for a sum type, i.e. a Rust enum.
    Sum {
        /// The size of the largest variant in bytes.
        max_variant_size: usize,
        variants: Vec<Repr>,
        discr: Box<Repr>,
    },

    Array {
        elem: Box<Repr>,

        /// Number of elements.
        length: usize,
    },

    Pointer {
        target: Box<Repr>,
    }
}

impl Memory {
    pub fn new() -> Self {
        Memory { next_id: 0, alloc_map: HashMap::new() }
    }

    pub fn allocate(&mut self, size: usize) -> Pointer {
        let id = AllocId(self.next_id);
        let alloc = Allocation { bytes: vec![0; size], relocations: BTreeMap::new() };
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

    fn get_bytes(&self, ptr: Pointer, size: usize) -> EvalResult<&[u8]> {
        let alloc = try!(self.get(ptr.alloc_id));
        try!(alloc.check_no_relocations(ptr.offset, ptr.offset + size));
        Ok(&alloc.bytes[ptr.offset..ptr.offset + size])
    }

    fn get_bytes_mut(&mut self, ptr: Pointer, size: usize) -> EvalResult<&mut [u8]> {
        let alloc = try!(self.get_mut(ptr.alloc_id));
        try!(alloc.check_no_relocations(ptr.offset, ptr.offset + size));
        Ok(&mut alloc.bytes[ptr.offset..ptr.offset + size])
    }

    pub fn copy(&mut self, src: Pointer, dest: Pointer, size: usize) -> EvalResult<()> {
        let (src_bytes, relocations) = {
            let alloc = try!(self.get_mut(src.alloc_id));
            try!(alloc.check_relocation_edges(src.offset, src.offset + size));
            let bytes = alloc.bytes[src.offset..src.offset + size].as_mut_ptr();

            let mut relocations: Vec<(usize, AllocId)> = alloc.relocations
                .range(Included(&src.offset), Excluded(&(src.offset + size)))
                .map(|(&k, &v)| (k, v))
                .collect();

            for &mut (ref mut offset, _) in &mut relocations {
                alloc.relocations.remove(offset);
                *offset += dest.offset - src.offset;
            }

            (bytes, relocations)
        };

        let dest_bytes = try!(self.get_bytes_mut(dest, size)).as_mut_ptr();

        // TODO(tsion): Clear the destination range's existing relocations.
        try!(self.get_mut(dest.alloc_id)).relocations.extend(relocations);

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

        Ok(())
    }

    pub fn read_ptr(&self, ptr: Pointer) -> EvalResult<Pointer> {
        let alloc = try!(self.get(ptr.alloc_id));
        try!(alloc.check_relocation_edges(ptr.offset, ptr.offset + POINTER_SIZE));
        let bytes = &alloc.bytes[ptr.offset..ptr.offset + POINTER_SIZE];
        let offset = byteorder::NativeEndian::read_u64(bytes) as usize;

        // TODO(tsion): Return an EvalError here instead of panicking.
        let alloc_id = *alloc.relocations.get(&ptr.offset).unwrap();

        Ok(Pointer { alloc_id: alloc_id, offset: offset })
    }

    // TODO(tsion): Detect invalid writes here and elsewhere.
    pub fn write_ptr(&mut self, dest: Pointer, ptr_val: Pointer) -> EvalResult<()> {
        {
            let bytes = try!(self.get_bytes_mut(dest, POINTER_SIZE));
            byteorder::NativeEndian::write_u64(bytes, ptr_val.offset as u64);
        }
        let alloc = try!(self.get_mut(dest.alloc_id));
        alloc.relocations.insert(dest.offset, ptr_val.alloc_id);
        Ok(())
    }

    pub fn read_primval(&self, ptr: Pointer, repr: &Repr) -> EvalResult<PrimVal> {
        match *repr {
            Repr::Bool => self.read_bool(ptr).map(PrimVal::Bool),
            Repr::I8   => self.read_i8(ptr).map(PrimVal::I8),
            Repr::I16  => self.read_i16(ptr).map(PrimVal::I16),
            Repr::I32  => self.read_i32(ptr).map(PrimVal::I32),
            Repr::I64  => self.read_i64(ptr).map(PrimVal::I64),
            Repr::U8   => self.read_u8(ptr).map(PrimVal::U8),
            Repr::U16  => self.read_u16(ptr).map(PrimVal::U16),
            Repr::U32  => self.read_u32(ptr).map(PrimVal::U32),
            Repr::U64  => self.read_u64(ptr).map(PrimVal::U64),
            _ => panic!("primitive read of non-primitive: {:?}", repr),
        }
    }

    pub fn write_primval(&mut self, ptr: Pointer, val: PrimVal) -> EvalResult<()> {
        match val {
            PrimVal::Bool(b) => self.write_bool(ptr, b),
            PrimVal::I8(n)   => self.write_i8(ptr, n),
            PrimVal::I16(n)  => self.write_i16(ptr, n),
            PrimVal::I32(n)  => self.write_i32(ptr, n),
            PrimVal::I64(n)  => self.write_i64(ptr, n),
            PrimVal::U8(n)   => self.write_u8(ptr, n),
            PrimVal::U16(n)  => self.write_u16(ptr, n),
            PrimVal::U32(n)  => self.write_u32(ptr, n),
            PrimVal::U64(n)  => self.write_u64(ptr, n),
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
        let bytes = try!(self.get_bytes_mut(ptr, 1));
        bytes[0] = b as u8;
        Ok(())
    }

    pub fn read_i8(&self, ptr: Pointer) -> EvalResult<i8> {
        self.get_bytes(ptr, 1).map(|b| b[0] as i8)
    }

    pub fn write_i8(&mut self, ptr: Pointer, n: i8) -> EvalResult<()> {
        self.get_bytes_mut(ptr, 1).map(|b| b[0] = n as u8)
    }

    pub fn read_i16(&self, ptr: Pointer) -> EvalResult<i16> {
        self.get_bytes(ptr, 2).map(byteorder::NativeEndian::read_i16)
    }

    pub fn write_i16(&mut self, ptr: Pointer, n: i16) -> EvalResult<()> {
        let bytes = try!(self.get_bytes_mut(ptr, 2));
        byteorder::NativeEndian::write_i16(bytes, n);
        Ok(())
    }

    pub fn read_i32(&self, ptr: Pointer) -> EvalResult<i32> {
        self.get_bytes(ptr, 4).map(byteorder::NativeEndian::read_i32)
    }

    pub fn write_i32(&mut self, ptr: Pointer, n: i32) -> EvalResult<()> {
        let bytes = try!(self.get_bytes_mut(ptr, 4));
        byteorder::NativeEndian::write_i32(bytes, n);
        Ok(())
    }

    pub fn read_i64(&self, ptr: Pointer) -> EvalResult<i64> {
        self.get_bytes(ptr, 8).map(byteorder::NativeEndian::read_i64)
    }

    pub fn write_i64(&mut self, ptr: Pointer, n: i64) -> EvalResult<()> {
        let bytes = try!(self.get_bytes_mut(ptr, 8));
        byteorder::NativeEndian::write_i64(bytes, n);
        Ok(())
    }

    pub fn read_u8(&self, ptr: Pointer) -> EvalResult<u8> {
        self.get_bytes(ptr, 1).map(|b| b[0] as u8)
    }

    pub fn write_u8(&mut self, ptr: Pointer, n: u8) -> EvalResult<()> {
        self.get_bytes_mut(ptr, 1).map(|b| b[0] = n as u8)
    }

    pub fn read_u16(&self, ptr: Pointer) -> EvalResult<u16> {
        self.get_bytes(ptr, 2).map(byteorder::NativeEndian::read_u16)
    }

    pub fn write_u16(&mut self, ptr: Pointer, n: u16) -> EvalResult<()> {
        let bytes = try!(self.get_bytes_mut(ptr, 2));
        byteorder::NativeEndian::write_u16(bytes, n);
        Ok(())
    }

    pub fn read_u32(&self, ptr: Pointer) -> EvalResult<u32> {
        self.get_bytes(ptr, 4).map(byteorder::NativeEndian::read_u32)
    }

    pub fn write_u32(&mut self, ptr: Pointer, n: u32) -> EvalResult<()> {
        let bytes = try!(self.get_bytes_mut(ptr, 4));
        byteorder::NativeEndian::write_u32(bytes, n);
        Ok(())
    }

    pub fn read_u64(&self, ptr: Pointer) -> EvalResult<u64> {
        self.get_bytes(ptr, 8).map(byteorder::NativeEndian::read_u64)
    }

    pub fn write_u64(&mut self, ptr: Pointer, n: u64) -> EvalResult<()> {
        let bytes = try!(self.get_bytes_mut(ptr, 8));
        byteorder::NativeEndian::write_u64(bytes, n);
        Ok(())
    }
}

impl Allocation {
    fn check_bounds(&self, start: usize, end: usize) -> EvalResult<()> {
        if start <= self.bytes.len() && end <= self.bytes.len() {
            Ok(())
        } else {
            Err(EvalError::PointerOutOfBounds)
        }
    }

    fn count_overlapping_relocations(&self, start: usize, end: usize) -> usize {
        self.relocations.range(
            Included(&start.saturating_sub(POINTER_SIZE - 1)),
            Excluded(&end)
        ).count()
    }

    fn check_relocation_edges(&self, start: usize, end: usize) -> EvalResult<()> {
        try!(self.check_bounds(start, end));
        let n =
            self.count_overlapping_relocations(start, start) +
            self.count_overlapping_relocations(end, end);
        if n == 0 {
            Ok(())
        } else {
            Err(EvalError::InvalidPointerAccess)
        }
    }

    fn check_no_relocations(&self, start: usize, end: usize) -> EvalResult<()> {
        try!(self.check_bounds(start, end));
        if self.count_overlapping_relocations(start, end) == 0 {
            Ok(())
        } else {
            Err(EvalError::InvalidPointerAccess)
        }
    }
}

impl Pointer {
    pub fn offset(self, i: usize) -> Self {
        Pointer { offset: self.offset + i, ..self }
    }
}

impl Repr {
    // TODO(tsion): Choice is based on host machine's type size. Should this be how miri works?
    pub fn isize() -> Self {
        match mem::size_of::<isize>() {
            4 => Repr::I32,
            8 => Repr::I64,
            _ => unimplemented!(),
        }
    }

    // TODO(tsion): Choice is based on host machine's type size. Should this be how miri works?
    pub fn usize() -> Self {
        match mem::size_of::<isize>() {
            4 => Repr::U32,
            8 => Repr::U64,
            _ => unimplemented!(),
        }
    }

    pub fn size(&self) -> usize {
        match *self {
            Repr::Bool => 1,
            Repr::I8  | Repr::U8  => 1,
            Repr::I16 | Repr::U16 => 2,
            Repr::I32 | Repr::U32 => 4,
            Repr::I64 | Repr::U64 => 8,
            Repr::Product { size, .. } => size,
            Repr::Sum { ref discr, max_variant_size, .. } => discr.size() + max_variant_size,
            Repr::Array { ref elem, length } => elem.size() * length,
            Repr::Pointer { .. } => POINTER_SIZE,
        }
    }
}
