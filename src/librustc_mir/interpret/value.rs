#![allow(unknown_lints)]
#![allow(float_cmp)]

use error::{EvalError, EvalResult};
use memory::{Memory, MemoryPointer, HasMemory, PointerArithmetic};
use rustc::ty::layout::HasDataLayout;

pub(super) fn bytes_to_f32(bytes: u128) -> f32 {
    f32::from_bits(bytes as u32)
}

pub(super) fn bytes_to_f64(bytes: u128) -> f64 {
    f64::from_bits(bytes as u64)
}

pub(super) fn f32_to_bytes(f: f32) -> u128 {
    f.to_bits() as u128
}

pub(super) fn f64_to_bytes(f: f64) -> u128 {
    f.to_bits() as u128
}

/// A `Value` represents a single self-contained Rust value.
///
/// A `Value` can either refer to a block of memory inside an allocation (`ByRef`) or to a primitve
/// value held directly, outside of any allocation (`ByVal`).  For `ByRef`-values, we remember
/// whether the pointer is supposed to be aligned or not (also see Lvalue).
///
/// For optimization of a few very common cases, there is also a representation for a pair of
/// primitive values (`ByValPair`). It allows Miri to avoid making allocations for checked binary
/// operations and fat pointers. This idea was taken from rustc's trans.
#[derive(Clone, Copy, Debug)]
pub enum Value {
    ByRef(Pointer, bool),
    ByVal(PrimVal),
    ByValPair(PrimVal, PrimVal),
}

/// A wrapper type around `PrimVal` that cannot be turned back into a `PrimVal` accidentally.
/// This type clears up a few APIs where having a `PrimVal` argument for something that is
/// potentially an integer pointer or a pointer to an allocation was unclear.
///
/// I (@oli-obk) believe it is less easy to mix up generic primvals and primvals that are just
/// the representation of pointers. Also all the sites that convert between primvals and pointers
/// are explicit now (and rare!)
#[derive(Clone, Copy, Debug)]
pub struct Pointer {
    primval: PrimVal,
}

impl<'tcx> Pointer {
    pub fn null() -> Self {
        PrimVal::Bytes(0).into()
    }
    pub fn to_ptr(self) -> EvalResult<'tcx, MemoryPointer> {
        self.primval.to_ptr()
    }
    pub fn into_inner_primval(self) -> PrimVal {
        self.primval
    }

    pub(crate) fn signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> EvalResult<'tcx, Self> {
        let layout = cx.data_layout();
        match self.primval {
            PrimVal::Bytes(b) => {
                assert_eq!(b as u64 as u128, b);
                Ok(Pointer::from(PrimVal::Bytes(layout.signed_offset(b as u64, i)? as u128)))
            },
            PrimVal::Ptr(ptr) => ptr.signed_offset(i, layout).map(Pointer::from),
            PrimVal::Undef => Err(EvalError::ReadUndefBytes),
        }
    }

    pub(crate) fn offset<C: HasDataLayout>(self, i: u64, cx: C) -> EvalResult<'tcx, Self> {
        let layout = cx.data_layout();
        match self.primval {
            PrimVal::Bytes(b) => {
                assert_eq!(b as u64 as u128, b);
                Ok(Pointer::from(PrimVal::Bytes(layout.offset(b as u64, i)? as u128)))
            },
            PrimVal::Ptr(ptr) => ptr.offset(i, layout).map(Pointer::from),
            PrimVal::Undef => Err(EvalError::ReadUndefBytes),
        }
    }

    pub(crate) fn wrapping_signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> EvalResult<'tcx, Self> {
        let layout = cx.data_layout();
        match self.primval {
            PrimVal::Bytes(b) => {
                assert_eq!(b as u64 as u128, b);
                Ok(Pointer::from(PrimVal::Bytes(layout.wrapping_signed_offset(b as u64, i) as u128)))
            },
            PrimVal::Ptr(ptr) => Ok(Pointer::from(ptr.wrapping_signed_offset(i, layout))),
            PrimVal::Undef => Err(EvalError::ReadUndefBytes),
        }
    }

    pub fn is_null(self) -> EvalResult<'tcx, bool> {
        match self.primval {
            PrimVal::Bytes(b) => Ok(b == 0),
            PrimVal::Ptr(_) => Ok(false),
            PrimVal::Undef => Err(EvalError::ReadUndefBytes),
        }
    }

    pub fn to_value_with_len(self, len: u64) -> Value {
        Value::ByValPair(self.primval, PrimVal::from_u128(len as u128))
    }

    pub fn to_value_with_vtable(self, vtable: MemoryPointer) -> Value {
        Value::ByValPair(self.primval, PrimVal::Ptr(vtable))
    }

    pub fn to_value(self) -> Value {
        Value::ByVal(self.primval)
    }
}

impl ::std::convert::From<PrimVal> for Pointer {
    fn from(primval: PrimVal) -> Self {
        Pointer { primval }
    }
}

impl ::std::convert::From<MemoryPointer> for Pointer {
    fn from(ptr: MemoryPointer) -> Self {
        PrimVal::Ptr(ptr).into()
    }
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
    /// relocation and its associated offset together as a `MemoryPointer` here.
    Ptr(MemoryPointer),

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
    #[inline]
    pub(super) fn by_ref(ptr: Pointer) -> Self {
        Value::ByRef(ptr, true)
    }

    /// Convert the value into a pointer (or a pointer-sized integer).  If the value is a ByRef,
    /// this may have to perform a load.
    pub(super) fn into_ptr(&self, mem: &mut Memory<'a, 'tcx>) -> EvalResult<'tcx, Pointer> {
        use self::Value::*;
        match *self {
            ByRef(ptr, aligned) => {
                mem.read_maybe_aligned(aligned, |mem| mem.read_ptr(ptr.to_ptr()?) )
            },
            ByVal(ptr) | ByValPair(ptr, _) => Ok(ptr.into()),
        }
    }

    pub(super) fn into_ptr_vtable_pair(
        &self,
        mem: &mut Memory<'a, 'tcx>
    ) -> EvalResult<'tcx, (Pointer, MemoryPointer)> {
        use self::Value::*;
        match *self {
            ByRef(ref_ptr, aligned) => {
                mem.read_maybe_aligned(aligned, |mem| {
                    let ptr = mem.read_ptr(ref_ptr.to_ptr()?)?;
                    let vtable = mem.read_ptr(ref_ptr.offset(mem.pointer_size(), mem.layout)?.to_ptr()?)?;
                    Ok((ptr, vtable.to_ptr()?))
                })
            }

            ByValPair(ptr, vtable) => Ok((ptr.into(), vtable.to_ptr()?)),

            _ => bug!("expected ptr and vtable, got {:?}", self),
        }
    }

    pub(super) fn into_slice(&self, mem: &mut Memory<'a, 'tcx>) -> EvalResult<'tcx, (Pointer, u64)> {
        use self::Value::*;
        match *self {
            ByRef(ref_ptr, aligned) => {
                mem.write_maybe_aligned(aligned, |mem| {
                    let ptr = mem.read_ptr(ref_ptr.to_ptr()?)?;
                    let len = mem.read_usize(ref_ptr.offset(mem.pointer_size(), mem.layout)?.to_ptr()?)?;
                    Ok((ptr, len))
                })
            },
            ByValPair(ptr, val) => {
                let len = val.to_u128()?;
                assert_eq!(len as u64 as u128, len);
                Ok((ptr.into(), len as u64))
            },
            ByVal(_) => bug!("expected ptr and length, got {:?}", self),
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
            PrimVal::Ptr(_) => Err(EvalError::ReadPointerAsBytes),
            PrimVal::Undef => Err(EvalError::ReadUndefBytes),
        }
    }

    pub fn to_ptr(self) -> EvalResult<'tcx, MemoryPointer> {
        match self {
            PrimVal::Bytes(_) => Err(EvalError::ReadBytesAsPointer),
            PrimVal::Ptr(p) => Ok(p),
            PrimVal::Undef => Err(EvalError::ReadUndefBytes),
        }
    }

    pub fn is_bytes(self) -> bool {
        match self {
            PrimVal::Bytes(_) => true,
            _ => false,
        }
    }

    pub fn is_ptr(self) -> bool {
        match self {
            PrimVal::Ptr(_) => true,
            _ => false,
        }
    }

    pub fn is_undef(self) -> bool {
        match self {
            PrimVal::Undef => true,
            _ => false,
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

     pub fn is_float(self) -> bool {
        use self::PrimValKind::*;
        match self {
            F32 | F64 => true,
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

    pub fn is_ptr(self) -> bool {
        use self::PrimValKind::*;
        match self {
            Ptr | FnPtr => true,
            _ => false,
        }
    }
}
