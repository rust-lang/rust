#![allow(unknown_lints)]
#![allow(float_cmp)]

use std::mem::transmute;

use error::{EvalError, EvalResult};
use memory::{AllocId, Memory, Pointer};

pub(super) fn bits_to_f32(bits: u64) -> f32 {
    unsafe { transmute::<u32, f32>(bits as u32) }
}

pub(super) fn bits_to_f64(bits: u64) -> f64 {
    unsafe { transmute::<u64, f64>(bits) }
}

pub(super) fn f32_to_bits(f: f32) -> u64 {
    unsafe { transmute::<f32, u32>(f) as u64 }
}

pub(super) fn f64_to_bits(f: f64) -> u64 {
    unsafe { transmute::<f64, u64>(f) }
}

pub(super) fn bits_to_bool(n: u64) -> bool {
    // FIXME(solson): Can we reach here due to user error?
    debug_assert!(n == 0 || n == 1, "bits interpreted as bool were {}", n);
    n & 1 == 1
}

/// A `Value` represents a single self-contained Rust value.
///
/// A `Value` can either refer to a block of memory inside an allocation (`ByRef`) or to a primitve
/// value held directly, outside of any allocation (`ByVal`).
///
/// For optimization of a few very common cases, there is also a representation for a pair of
/// primitive values (`ByValPair`). It allows Miri to avoid making allocations for checked binary
/// operations and fat pointers. This idea was taken from rustc's trans.
#[derive(Clone, Copy, Debug)]
pub enum Value {
    ByRef(Pointer),
    ByVal(PrimVal),
    ByValPair(PrimVal, PrimVal),
}

/// A `PrimVal` represents an immediate, primitive value existing outside of an allocation. It is
/// considered to be like a
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrimVal {
    Bytes(u64),

    // FIXME(solson): Rename this variant to Ptr.
    // FIXME(solson): Outdated comment, pulled from `relocations` field I deleted.
    /// This field is initialized when the `PrimVal` represents a pointer into an `Allocation`. An
    /// `Allocation` in the `memory` module has a list of relocations, but a `PrimVal` is only
    /// large enough to contain one, hence the `Option`.
    Pointer(Pointer),

    Undefined,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrimValKind {
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F32, F64,
    Bool,
    Char,
    Ptr,
    FnPtr,
}

impl<'a, 'tcx: 'a> Value {
    pub(super) fn read_ptr(&self, mem: &Memory<'a, 'tcx>) -> EvalResult<'tcx, Pointer> {
        use self::Value::*;
        match *self {
            ByRef(ptr) => mem.read_ptr(ptr),
            ByVal(ptr) | ByValPair(ptr, _) => Ok(ptr.to_ptr()),
        }
    }

    pub(super) fn expect_ptr_vtable_pair(
        &self,
        mem: &Memory<'a, 'tcx>
    ) -> EvalResult<'tcx, (Pointer, Pointer)> {
        use self::Value::*;
        match *self {
            ByRef(ref_ptr) => {
                let ptr = mem.read_ptr(ref_ptr)?;
                let vtable = mem.read_ptr(ref_ptr.offset(mem.pointer_size()))?;
                Ok((ptr, vtable))
            }

            ByValPair(ptr, vtable) => Ok((ptr.to_ptr(), vtable.to_ptr())),

            _ => bug!("expected ptr and vtable, got {:?}", self),
        }
    }

    pub(super) fn expect_slice(&self, mem: &Memory<'a, 'tcx>) -> EvalResult<'tcx, (Pointer, u64)> {
        use self::Value::*;
        match *self {
            ByRef(ref_ptr) => {
                let ptr = mem.read_ptr(ref_ptr)?;
                let len = mem.read_usize(ref_ptr.offset(mem.pointer_size()))?;
                Ok((ptr, len))
            },
            ByValPair(ptr, val) => {
                Ok((ptr.to_ptr(), val.try_as_uint()?))
            },
            _ => unimplemented!(),
        }
    }
}

impl PrimVal {
    // FIXME(solson): Remove this. It's a temporary function to aid refactoring, but it shouldn't
    // stick around with this name.
    pub fn bits(&self) -> u64 {
        match *self {
            PrimVal::Bytes(b) => b,
            PrimVal::Pointer(p) => p.offset,
            PrimVal::Undefined => panic!(".bits()() on PrimVal::Undefined"),
        }
    }

    // FIXME(solson): Remove this. It's a temporary function to aid refactoring, but it shouldn't
    // stick around with this name.
    pub fn relocation(&self) -> Option<AllocId> {
        if let PrimVal::Pointer(ref p) = *self {
            Some(p.alloc_id)
        } else {
            None
        }
    }

    pub fn from_bool(b: bool) -> Self {
        PrimVal::Bytes(b as u64)
    }

    pub fn from_char(c: char) -> Self {
        PrimVal::Bytes(c as u64)
    }

    pub fn from_f32(f: f32) -> Self {
        PrimVal::Bytes(f32_to_bits(f))
    }

    pub fn from_f64(f: f64) -> Self {
        PrimVal::Bytes(f64_to_bits(f))
    }

    pub fn from_uint(n: u64) -> Self {
        PrimVal::Bytes(n)
    }

    pub fn from_int(n: i64) -> Self {
        PrimVal::Bytes(n as u64)
    }

    pub fn to_f32(self) -> f32 {
        assert!(self.relocation().is_none());
        bits_to_f32(self.bits())
    }

    pub fn to_f64(self) -> f64 {
        assert!(self.relocation().is_none());
        bits_to_f64(self.bits())
    }

    pub fn to_ptr(self) -> Pointer {
        self.relocation().map(|alloc_id| {
            Pointer::new(alloc_id, self.bits())
        }).unwrap_or_else(|| Pointer::from_int(self.bits()))
    }

    pub fn try_as_uint<'tcx>(self) -> EvalResult<'tcx, u64> {
        self.to_ptr().to_int()
    }

    pub fn to_u64(self) -> u64 {
        if let Some(ptr) = self.try_as_ptr() {
            return ptr.to_int().expect("non abstract ptr") as u64;
        }
        self.bits()
    }

    pub fn to_i64(self) -> i64 {
        if let Some(ptr) = self.try_as_ptr() {
            return ptr.to_int().expect("non abstract ptr") as i64;
        }
        self.bits() as i64
    }

    pub fn try_as_ptr(self) -> Option<Pointer> {
        self.relocation().map(|alloc_id| {
            Pointer::new(alloc_id, self.bits())
        })
    }

    pub fn try_as_bool<'tcx>(self) -> EvalResult<'tcx, bool> {
        match self.bits() {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(EvalError::InvalidBool),
        }
    }
}

impl PrimValKind {
    pub fn is_int(self) -> bool {
        use self::PrimValKind::*;
        match self {
            I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 => true,
            _ => false,
        }
    }

    pub fn from_uint_size(size: u64) -> Self {
        match size {
            1 => PrimValKind::U8,
            2 => PrimValKind::U16,
            4 => PrimValKind::U32,
            8 => PrimValKind::U64,
            _ => bug!("can't make uint with size {}", size),
        }
    }

    pub fn from_int_size(size: u64) -> Self {
        match size {
            1 => PrimValKind::I8,
            2 => PrimValKind::I16,
            4 => PrimValKind::I32,
            8 => PrimValKind::I64,
            _ => bug!("can't make int with size {}", size),
        }
    }
}
