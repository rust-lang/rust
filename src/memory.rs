use byteorder::{self, ByteOrder};
use std::collections::HashMap;
use std::mem;
use std::ptr;

use interpreter::{EvalError, EvalResult};

pub struct Memory {
    next_id: u64,
    alloc_map: HashMap<u64, Allocation>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AllocId(u64);

#[derive(Debug)]
pub struct Allocation {
    pub bytes: Vec<u8>,
    // TODO(tsion): relocations
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
    Int,
    Aggregate {
        size: usize,
        fields: Vec<FieldRepr>,
    },
}

impl Memory {
    pub fn new() -> Self {
        Memory { next_id: 0, alloc_map: HashMap::new() }
    }

    pub fn allocate(&mut self, size: usize) -> Pointer {
        let id = AllocId(self.next_id);
        let alloc = Allocation { bytes: vec![0; size] };
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
        try!(alloc.check_bytes(ptr.offset, ptr.offset + size));
        Ok(&alloc.bytes[ptr.offset..ptr.offset + size])
    }

    fn get_bytes_mut(&mut self, ptr: Pointer, size: usize) -> EvalResult<&mut [u8]> {
        let alloc = try!(self.get_mut(ptr.alloc_id));
        try!(alloc.check_bytes(ptr.offset, ptr.offset + size));
        Ok(&mut alloc.bytes[ptr.offset..ptr.offset + size])
    }

    pub fn copy(&mut self, src: Pointer, dest: Pointer, size: usize) -> EvalResult<()> {
        let src_bytes = try!(self.get_bytes_mut(src, size)).as_mut_ptr();
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

        Ok(())
    }

    pub fn read_int(&self, ptr: Pointer) -> EvalResult<i64> {
        self.get_bytes(ptr, Repr::Int.size()).map(byteorder::NativeEndian::read_i64)
    }

    pub fn write_int(&mut self, ptr: Pointer, n: i64) -> EvalResult<()> {
        let bytes = try!(self.get_bytes_mut(ptr, Repr::Int.size()));
        byteorder::NativeEndian::write_i64(bytes, n);
        Ok(())
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
}

impl Allocation {
    fn check_bytes(&self, start: usize, end: usize) -> EvalResult<()> {
        if start <= self.bytes.len() && end <= self.bytes.len() {
            Ok(())
        } else {
            Err(EvalError::PointerOutOfBounds)
        }
    }
}

impl Pointer {
    pub fn offset(self, i: usize) -> Self {
        // TODO(tsion): Check for offset out of bounds.
        Pointer { offset: self.offset + i, ..self }
    }
}

impl Repr {
    pub fn size(&self) -> usize {
        match *self {
            Repr::Bool => 1,
            Repr::Int => mem::size_of::<i64>(),
            Repr::Aggregate { size, .. } => size,
        }
    }
}
