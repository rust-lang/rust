#![allow(unknown_lints)]
#![allow(float_cmp)]

use std::mem::transmute;

use error::{EvalError, EvalResult};
use memory::{Memory, Pointer};

pub(super) fn bytes_to_f32(bytes: u128) -> f32 {
    unsafe { transmute::<u32, f32>(bytes as u32) }
}

pub(super) fn bytes_to_f64(bytes: u128) -> f64 {
    unsafe { transmute::<u64, f64>(bytes as u64) }
}

pub(super) fn f32_to_bytes(f: f32) -> u128 {
    unsafe { transmute::<f32, u32>(f) as u128 }
}

pub(super) fn f64_to_bytes(f: f64) -> u128 {
    unsafe { transmute::<f64, u64>(f) as u128 }
}

pub(super) fn bytes_to_bool(n: u128) -> bool {
    // FIXME(solson): Can we reach here due to user error?
    debug_assert!(n == 0 || n == 1, "bytes interpreted as bool were {}", n);
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

/// A `PrimVal` represents an immediate, primitive value existing outside of a
/// `memory::Allocation`. It is in many ways like a small chunk of a `Allocation`, up to 8 bytes in
/// size. Like a range of bytes in an `Allocation`, a `PrimVal` can either represent the raw bytes
/// of a simple value, a pointer into another `Allocation`, or be undefined.
#[derive(Clone, Copy, Debug)]
pub enum PrimVal {
    /// The raw bytes of a simple value.
    Bytes(u128),

    /// A pointer into an `Allocation`. An `Allocation` in the `memory` module has a list of
    /// relocations, but a `PrimVal` is only large enough to contain one, so we just represent the
    /// relocation and its associated offset together as a `Pointer` here.
    Ptr(Pointer),

    /// An undefined `PrimVal`, for representing values that aren't safe to examine, but are safe
    /// to copy around, just like undefined bytes in an `Allocation`.
    Undef,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrimValKind {
    I8, I16, I32, I64, I128,
    U8, U16, U32, U64, U128,
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
            ByVal(ptr) | ByValPair(ptr, _) => ptr.to_ptr(),
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

            ByValPair(ptr, vtable) => Ok((ptr.to_ptr()?, vtable.to_ptr()?)),

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
                let len = val.to_u128()?;
                assert_eq!(len as u64 as u128, len);
                Ok((ptr.to_ptr()?, len as u64))
            },
            _ => unimplemented!(),
        }
    }
}

impl<'tcx> PrimVal {
    pub fn from_u128(n: u128) -> Self {
        PrimVal::Bytes(n)
    }

    pub fn from_i128(n: i128) -> Self {
        PrimVal::Bytes(n as u128)
    }

    pub fn from_f32(f: f32) -> Self {
        PrimVal::Bytes(f32_to_bytes(f))
    }

    pub fn from_f64(f: f64) -> Self {
        PrimVal::Bytes(f64_to_bytes(f))
    }

    pub fn from_bool(b: bool) -> Self {
        PrimVal::Bytes(b as u128)
    }

    pub fn from_char(c: char) -> Self {
        PrimVal::Bytes(c as u128)
    }

    pub fn to_bytes(self) -> EvalResult<'tcx, u128> {
        match self {
            PrimVal::Bytes(b) => Ok(b),
            PrimVal::Ptr(p) => p.to_int().map(|b| b as u128),
            PrimVal::Undef => Err(EvalError::ReadUndefBytes),
        }
    }

    pub fn to_ptr(self) -> EvalResult<'tcx, Pointer> {
        match self {
            PrimVal::Bytes(b) => Ok(Pointer::from_int(b as u64)),
            PrimVal::Ptr(p) => Ok(p),
            PrimVal::Undef => Err(EvalError::ReadUndefBytes),
        }
    }

    pub fn to_u128(self) -> EvalResult<'tcx, u128> {
        self.to_bytes()
    }

    pub fn to_u64(self) -> EvalResult<'tcx, u64> {
        self.to_bytes().map(|b| {
            assert_eq!(b as u64 as u128, b);
            b as u64
        })
    }

    pub fn to_i32(self) -> EvalResult<'tcx, i32> {
        self.to_bytes().map(|b| {
            assert_eq!(b as i32 as u128, b);
            b as i32
        })
    }

    pub fn to_i128(self) -> EvalResult<'tcx, i128> {
        self.to_bytes().map(|b| b as i128)
    }

    pub fn to_i64(self) -> EvalResult<'tcx, i64> {
        self.to_bytes().map(|b| {
            assert_eq!(b as i64 as u128, b);
            b as i64
        })
    }

    pub fn to_f32(self) -> EvalResult<'tcx, f32> {
        self.to_bytes().map(bytes_to_f32)
    }

    pub fn to_f64(self) -> EvalResult<'tcx, f64> {
        self.to_bytes().map(bytes_to_f64)
    }

    pub fn to_bool(self) -> EvalResult<'tcx, bool> {
        match self.to_bytes()? {
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
            I8 | I16 | I32 | I64 | I128 | U8 | U16 | U32 | U64 | U128 => true,
            _ => false,
        }
    }

    pub fn is_signed_int(self) -> bool {
        use self::PrimValKind::*;
        match self {
            I8 | I16 | I32 | I64 | I128 => true,
            _ => false,
        }
    }

    pub fn from_uint_size(size: u64) -> Self {
        match size {
            1 => PrimValKind::U8,
            2 => PrimValKind::U16,
            4 => PrimValKind::U32,
            8 => PrimValKind::U64,
            16 => PrimValKind::U128,
            _ => bug!("can't make uint with size {}", size),
        }
    }

    pub fn from_int_size(size: u64) -> Self {
        match size {
            1 => PrimValKind::I8,
            2 => PrimValKind::I16,
            4 => PrimValKind::I32,
            8 => PrimValKind::I64,
            16 => PrimValKind::I128,
            _ => bug!("can't make int with size {}", size),
        }
    }
}
