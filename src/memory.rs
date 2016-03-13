use byteorder::{self, ByteOrder};
use std::collections::HashMap;
use std::ptr;

use interpreter::{EvalError, EvalResult};
use primval::PrimVal;

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
pub enum IntRepr { I8, I16, I32, I64 }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldRepr {
    pub offset: usize,
    pub repr: Repr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Repr {
    Bool,
    Int(IntRepr),

    /// The representation for product types including tuples, structs, and the contents of enum
    /// variants.
    Product {
        /// Size in bytes.
        size: usize,
        fields: Vec<FieldRepr>,
    },

    /// The representation for a sum type, i.e. a Rust enum.
    Sum {
        /// The size of the discriminant in bytes.
        discr_size: usize,

        /// The size of the largest variant in bytes.
        max_variant_size: usize,

        variants: Vec<Repr>,
    },

    // Array {
    //     /// Number of elements.
    //     length: usize,
    //     elem: Repr,
    // },
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

    pub fn read_primval(&self, ptr: Pointer, repr: &Repr) -> EvalResult<PrimVal> {
        match *repr {
            Repr::Bool => self.read_bool(ptr).map(PrimVal::Bool),
            Repr::Int(IntRepr::I8) => self.read_i8(ptr).map(PrimVal::I8),
            Repr::Int(IntRepr::I16) => self.read_i16(ptr).map(PrimVal::I16),
            Repr::Int(IntRepr::I32) => self.read_i32(ptr).map(PrimVal::I32),
            Repr::Int(IntRepr::I64) => self.read_i64(ptr).map(PrimVal::I64),
            _ => panic!("primitive read of non-primitive: {:?}", repr),
        }
    }

    pub fn write_primval(&mut self, ptr: Pointer, val: PrimVal) -> EvalResult<()> {
        match val {
            PrimVal::Bool(b) => self.write_bool(ptr, b),
            PrimVal::I8(n) => self.write_i8(ptr, n),
            PrimVal::I16(n) => self.write_i16(ptr, n),
            PrimVal::I32(n) => self.write_i32(ptr, n),
            PrimVal::I64(n) => self.write_i64(ptr, n),
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
            Repr::Int(IntRepr::I8) => 1,
            Repr::Int(IntRepr::I16) => 2,
            Repr::Int(IntRepr::I32) => 4,
            Repr::Int(IntRepr::I64) => 8,
            Repr::Product { size, .. } => size,
            Repr::Sum { discr_size, max_variant_size, .. } => discr_size + max_variant_size,
        }
    }
}
