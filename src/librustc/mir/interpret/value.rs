#![allow(unknown_lints)]

use ty::layout::{Align, HasDataLayout, Size};
use ty;

use super::{EvalResult, MemoryPointer, PointerArithmetic, Allocation};

/// Represents a constant value in Rust. ByVal and ScalarPair are optimizations which
/// matches Value's optimizations for easy conversions between these two types
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash)]
pub enum ConstValue<'tcx> {
    /// Used only for types with layout::abi::Scalar ABI and ZSTs which use Scalar::Undef
    Scalar(Scalar),
    /// Used only for types with layout::abi::ScalarPair
    ScalarPair(Scalar, Scalar),
    /// Used only for the remaining cases. An allocation + offset into the allocation
    ByRef(&'tcx Allocation, Size),
}

impl<'tcx> ConstValue<'tcx> {
    #[inline]
    pub fn from_byval_value(val: Value) -> Self {
        match val {
            Value::ByRef(..) => bug!(),
            Value::ScalarPair(a, b) => ConstValue::ScalarPair(a, b),
            Value::Scalar(val) => ConstValue::Scalar(val),
        }
    }

    #[inline]
    pub fn to_byval_value(&self) -> Option<Value> {
        match *self {
            ConstValue::ByRef(..) => None,
            ConstValue::ScalarPair(a, b) => Some(Value::ScalarPair(a, b)),
            ConstValue::Scalar(val) => Some(Value::Scalar(val)),
        }
    }

    #[inline]
    pub fn from_primval(val: Scalar) -> Self {
        ConstValue::Scalar(val)
    }

    #[inline]
    pub fn to_primval(&self) -> Option<Scalar> {
        match *self {
            ConstValue::ByRef(..) => None,
            ConstValue::ScalarPair(..) => None,
            ConstValue::Scalar(val) => Some(val),
        }
    }

    #[inline]
    pub fn to_bits(&self) -> Option<u128> {
        match self.to_primval() {
            Some(Scalar::Bytes(val)) => Some(val),
            _ => None,
        }
    }

    #[inline]
    pub fn to_ptr(&self) -> Option<MemoryPointer> {
        match self.to_primval() {
            Some(Scalar::Ptr(ptr)) => Some(ptr),
            _ => None,
        }
    }
}

/// A `Value` represents a single self-contained Rust value.
///
/// A `Value` can either refer to a block of memory inside an allocation (`ByRef`) or to a primitve
/// value held directly, outside of any allocation (`ByVal`).  For `ByRef`-values, we remember
/// whether the pointer is supposed to be aligned or not (also see Place).
///
/// For optimization of a few very common cases, there is also a representation for a pair of
/// primitive values (`ScalarPair`). It allows Miri to avoid making allocations for checked binary
/// operations and fat pointers. This idea was taken from rustc's codegen.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, RustcEncodable, RustcDecodable, Hash)]
pub enum Value {
    ByRef(Pointer, Align),
    Scalar(Scalar),
    ScalarPair(Scalar, Scalar),
}

impl<'tcx> ty::TypeFoldable<'tcx> for Value {
    fn super_fold_with<'gcx: 'tcx, F: ty::fold::TypeFolder<'gcx, 'tcx>>(&self, _: &mut F) -> Self {
        *self
    }
    fn super_visit_with<V: ty::fold::TypeVisitor<'tcx>>(&self, _: &mut V) -> bool {
        false
    }
}

/// A wrapper type around `Scalar` that cannot be turned back into a `Scalar` accidentally.
/// This type clears up a few APIs where having a `Scalar` argument for something that is
/// potentially an integer pointer or a pointer to an allocation was unclear.
///
/// I (@oli-obk) believe it is less easy to mix up generic primvals and primvals that are just
/// the representation of pointers. Also all the sites that convert between primvals and pointers
/// are explicit now (and rare!)
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, RustcEncodable, RustcDecodable, Hash)]
pub struct Pointer {
    pub primval: Scalar,
}

impl<'tcx> Pointer {
    pub fn null() -> Self {
        Scalar::Bytes(0).into()
    }
    pub fn to_ptr(self) -> EvalResult<'tcx, MemoryPointer> {
        self.primval.to_ptr()
    }
    pub fn into_inner_primval(self) -> Scalar {
        self.primval
    }

    pub fn signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> EvalResult<'tcx, Self> {
        let layout = cx.data_layout();
        match self.primval {
            Scalar::Bytes(b) => {
                assert_eq!(b as u64 as u128, b);
                Ok(Pointer::from(
                    Scalar::Bytes(layout.signed_offset(b as u64, i)? as u128),
                ))
            }
            Scalar::Ptr(ptr) => ptr.signed_offset(i, layout).map(Pointer::from),
            Scalar::Undef => err!(ReadUndefBytes),
        }
    }

    pub fn offset<C: HasDataLayout>(self, i: Size, cx: C) -> EvalResult<'tcx, Self> {
        let layout = cx.data_layout();
        match self.primval {
            Scalar::Bytes(b) => {
                assert_eq!(b as u64 as u128, b);
                Ok(Pointer::from(
                    Scalar::Bytes(layout.offset(b as u64, i.bytes())? as u128),
                ))
            }
            Scalar::Ptr(ptr) => ptr.offset(i, layout).map(Pointer::from),
            Scalar::Undef => err!(ReadUndefBytes),
        }
    }

    pub fn wrapping_signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> EvalResult<'tcx, Self> {
        let layout = cx.data_layout();
        match self.primval {
            Scalar::Bytes(b) => {
                assert_eq!(b as u64 as u128, b);
                Ok(Pointer::from(Scalar::Bytes(
                    layout.wrapping_signed_offset(b as u64, i) as u128,
                )))
            }
            Scalar::Ptr(ptr) => Ok(Pointer::from(ptr.wrapping_signed_offset(i, layout))),
            Scalar::Undef => err!(ReadUndefBytes),
        }
    }

    pub fn is_null(self) -> EvalResult<'tcx, bool> {
        match self.primval {
            Scalar::Bytes(b) => Ok(b == 0),
            Scalar::Ptr(_) => Ok(false),
            Scalar::Undef => err!(ReadUndefBytes),
        }
    }

    pub fn to_value_with_len(self, len: u64) -> Value {
        Value::ScalarPair(self.primval, Scalar::from_u128(len as u128))
    }

    pub fn to_value_with_vtable(self, vtable: MemoryPointer) -> Value {
        Value::ScalarPair(self.primval, Scalar::Ptr(vtable))
    }

    pub fn to_value(self) -> Value {
        Value::Scalar(self.primval)
    }
}

impl ::std::convert::From<Scalar> for Pointer {
    fn from(primval: Scalar) -> Self {
        Pointer { primval }
    }
}

impl ::std::convert::From<MemoryPointer> for Pointer {
    fn from(ptr: MemoryPointer) -> Self {
        Scalar::Ptr(ptr).into()
    }
}

/// A `Scalar` represents an immediate, primitive value existing outside of a
/// `memory::Allocation`. It is in many ways like a small chunk of a `Allocation`, up to 8 bytes in
/// size. Like a range of bytes in an `Allocation`, a `Scalar` can either represent the raw bytes
/// of a simple value, a pointer into another `Allocation`, or be undefined.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, RustcEncodable, RustcDecodable, Hash)]
pub enum Scalar {
    /// The raw bytes of a simple value.
    Bytes(u128),

    /// A pointer into an `Allocation`. An `Allocation` in the `memory` module has a list of
    /// relocations, but a `Scalar` is only large enough to contain one, so we just represent the
    /// relocation and its associated offset together as a `MemoryPointer` here.
    Ptr(MemoryPointer),

    /// An undefined `Scalar`, for representing values that aren't safe to examine, but are safe
    /// to copy around, just like undefined bytes in an `Allocation`.
    Undef,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarKind {
    I8, I16, I32, I64, I128,
    U8, U16, U32, U64, U128,
    F32, F64,
    Ptr, FnPtr,
    Bool,
    Char,
}

impl<'tcx> Scalar {
    pub fn from_u128(n: u128) -> Self {
        Scalar::Bytes(n)
    }

    pub fn from_i128(n: i128) -> Self {
        Scalar::Bytes(n as u128)
    }

    pub fn from_bool(b: bool) -> Self {
        Scalar::Bytes(b as u128)
    }

    pub fn from_char(c: char) -> Self {
        Scalar::Bytes(c as u128)
    }

    pub fn to_bytes(self) -> EvalResult<'tcx, u128> {
        match self {
            Scalar::Bytes(b) => Ok(b),
            Scalar::Ptr(_) => err!(ReadPointerAsBytes),
            Scalar::Undef => err!(ReadUndefBytes),
        }
    }

    pub fn to_ptr(self) -> EvalResult<'tcx, MemoryPointer> {
        match self {
            Scalar::Bytes(_) => err!(ReadBytesAsPointer),
            Scalar::Ptr(p) => Ok(p),
            Scalar::Undef => err!(ReadUndefBytes),
        }
    }

    pub fn is_bytes(self) -> bool {
        match self {
            Scalar::Bytes(_) => true,
            _ => false,
        }
    }

    pub fn is_ptr(self) -> bool {
        match self {
            Scalar::Ptr(_) => true,
            _ => false,
        }
    }

    pub fn is_undef(self) -> bool {
        match self {
            Scalar::Undef => true,
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

    pub fn to_bool(self) -> EvalResult<'tcx, bool> {
        match self.to_bytes()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => err!(InvalidBool),
        }
    }
}

impl ScalarKind {
    pub fn is_int(self) -> bool {
        use self::ScalarKind::*;
        match self {
            I8 | I16 | I32 | I64 | I128 | U8 | U16 | U32 | U64 | U128 => true,
            _ => false,
        }
    }

    pub fn is_signed_int(self) -> bool {
        use self::ScalarKind::*;
        match self {
            I8 | I16 | I32 | I64 | I128 => true,
            _ => false,
        }
    }

    pub fn is_float(self) -> bool {
        use self::ScalarKind::*;
        match self {
            F32 | F64 => true,
            _ => false,
        }
    }

    pub fn from_uint_size(size: Size) -> Self {
        match size.bytes() {
            1 => ScalarKind::U8,
            2 => ScalarKind::U16,
            4 => ScalarKind::U32,
            8 => ScalarKind::U64,
            16 => ScalarKind::U128,
            _ => bug!("can't make uint with size {}", size.bytes()),
        }
    }

    pub fn from_int_size(size: Size) -> Self {
        match size.bytes() {
            1 => ScalarKind::I8,
            2 => ScalarKind::I16,
            4 => ScalarKind::I32,
            8 => ScalarKind::I64,
            16 => ScalarKind::I128,
            _ => bug!("can't make int with size {}", size.bytes()),
        }
    }

    pub fn is_ptr(self) -> bool {
        use self::ScalarKind::*;
        match self {
            Ptr | FnPtr => true,
            _ => false,
        }
    }
}
