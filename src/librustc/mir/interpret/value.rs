#![allow(unknown_lints)]

use ty::layout::{Align, HasDataLayout, Size};
use ty;
use ty::subst::Substs;
use hir::def_id::DefId;

use super::{EvalResult, Pointer, PointerArithmetic, Allocation};

/// Represents a constant value in Rust. Scalar and ScalarPair are optimizations which
/// matches Value's optimizations for easy conversions between these two types
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash)]
pub enum ConstValue<'tcx> {
    /// Never returned from the `const_eval` query, but the HIR contains these frequently in order
    /// to allow HIR creation to happen for everything before needing to be able to run constant
    /// evaluation
    Unevaluated(DefId, &'tcx Substs<'tcx>),
    /// Used only for types with layout::abi::Scalar ABI and ZSTs
    Scalar(Scalar),
    /// Used only for types with layout::abi::ScalarPair
    ScalarPair(Scalar, Scalar),
    /// Used only for the remaining cases. An allocation + offset into the allocation
    ByRef(&'tcx Allocation, Size),
}

impl<'tcx> ConstValue<'tcx> {
    #[inline]
    pub fn from_byval_value(val: Value) -> EvalResult<'static, Self> {
        Ok(match val {
            Value::ByRef(..) => bug!(),
            Value::ScalarPair(a, b) => ConstValue::ScalarPair(
                a.unwrap_or_err()?,
                b.unwrap_or_err()?,
            ),
            Value::Scalar(val) => ConstValue::Scalar(val.unwrap_or_err()?),
        })
    }

    #[inline]
    pub fn to_byval_value(&self) -> Option<Value> {
        match *self {
            ConstValue::Unevaluated(..) |
            ConstValue::ByRef(..) => None,
            ConstValue::ScalarPair(a, b) => Some(Value::ScalarPair(a.into(), b.into())),
            ConstValue::Scalar(val) => Some(Value::Scalar(val.into())),
        }
    }

    #[inline]
    pub fn try_to_scalar(&self) -> Option<Scalar> {
        match *self {
            ConstValue::Unevaluated(..) |
            ConstValue::ByRef(..) |
            ConstValue::ScalarPair(..) => None,
            ConstValue::Scalar(val) => Some(val),
        }
    }

    #[inline]
    pub fn to_bits(&self, size: Size) -> Option<u128> {
        self.try_to_scalar()?.to_bits(size).ok()
    }

    #[inline]
    pub fn to_ptr(&self) -> Option<Pointer> {
        self.try_to_scalar()?.to_ptr().ok()
    }
}

/// A `Value` represents a single self-contained Rust value.
///
/// A `Value` can either refer to a block of memory inside an allocation (`ByRef`) or to a primitve
/// value held directly, outside of any allocation (`Scalar`).  For `ByRef`-values, we remember
/// whether the pointer is supposed to be aligned or not (also see Place).
///
/// For optimization of a few very common cases, there is also a representation for a pair of
/// primitive values (`ScalarPair`). It allows Miri to avoid making allocations for checked binary
/// operations and fat pointers. This idea was taken from rustc's codegen.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, RustcEncodable, RustcDecodable, Hash)]
pub enum Value {
    ByRef(Scalar, Align),
    Scalar(ScalarMaybeUndef),
    ScalarPair(ScalarMaybeUndef, ScalarMaybeUndef),
}

impl<'tcx> ty::TypeFoldable<'tcx> for Value {
    fn super_fold_with<'gcx: 'tcx, F: ty::fold::TypeFolder<'gcx, 'tcx>>(&self, _: &mut F) -> Self {
        *self
    }
    fn super_visit_with<V: ty::fold::TypeVisitor<'tcx>>(&self, _: &mut V) -> bool {
        false
    }
}

impl<'tcx> Scalar {
    pub fn ptr_null<C: HasDataLayout>(cx: C) -> Self {
        Scalar::Bits {
            bits: 0,
            size: cx.data_layout().pointer_size.bytes() as u8,
        }
    }

    pub fn to_value_with_len<C: HasDataLayout>(self, len: u64, cx: C) -> Value {
        ScalarMaybeUndef::Scalar(self).to_value_with_len(len, cx)
    }

    pub fn ptr_signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> EvalResult<'tcx, Self> {
        let layout = cx.data_layout();
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, layout.pointer_size.bytes());
                Ok(Scalar::Bits {
                    bits: layout.signed_offset(bits as u64, i)? as u128,
                    size,
                })
            }
            Scalar::Ptr(ptr) => ptr.signed_offset(i, layout).map(Scalar::Ptr),
        }
    }

    pub fn ptr_offset<C: HasDataLayout>(self, i: Size, cx: C) -> EvalResult<'tcx, Self> {
        let layout = cx.data_layout();
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, layout.pointer_size.bytes());
                Ok(Scalar::Bits {
                    bits: layout.offset(bits as u64, i.bytes())? as u128,
                    size,
                })
            }
            Scalar::Ptr(ptr) => ptr.offset(i, layout).map(Scalar::Ptr),
        }
    }

    pub fn ptr_wrapping_signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> Self {
        let layout = cx.data_layout();
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(size as u64, layout.pointer_size.bytes());
                Scalar::Bits {
                    bits: layout.wrapping_signed_offset(bits as u64, i) as u128,
                    size,
                }
            }
            Scalar::Ptr(ptr) => Scalar::Ptr(ptr.wrapping_signed_offset(i, layout)),
        }
    }

    pub fn is_null_ptr<C: HasDataLayout>(self, cx: C) -> bool {
        match self {
            Scalar::Bits { bits, size } =>  {
                assert_eq!(size as u64, cx.data_layout().pointer_size.bytes());
                bits == 0
            },
            Scalar::Ptr(_) => false,
        }
    }

    pub fn to_value(self) -> Value {
        Value::Scalar(ScalarMaybeUndef::Scalar(self))
    }
}

impl From<Pointer> for Scalar {
    fn from(ptr: Pointer) -> Self {
        Scalar::Ptr(ptr)
    }
}

/// A `Scalar` represents an immediate, primitive value existing outside of a
/// `memory::Allocation`. It is in many ways like a small chunk of a `Allocation`, up to 8 bytes in
/// size. Like a range of bytes in an `Allocation`, a `Scalar` can either represent the raw bytes
/// of a simple value or a pointer into another `Allocation`
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, RustcEncodable, RustcDecodable, Hash)]
pub enum Scalar {
    /// The raw bytes of a simple value.
    Bits {
        /// The first `size` bytes are the value.
        /// Do not try to read less or more bytes that that
        size: u8,
        bits: u128,
    },

    /// A pointer into an `Allocation`. An `Allocation` in the `memory` module has a list of
    /// relocations, but a `Scalar` is only large enough to contain one, so we just represent the
    /// relocation and its associated offset together as a `Pointer` here.
    Ptr(Pointer),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, RustcEncodable, RustcDecodable, Hash)]
pub enum ScalarMaybeUndef {
    Scalar(Scalar),
    Undef,
}

impl From<Scalar> for ScalarMaybeUndef {
    fn from(s: Scalar) -> Self {
        ScalarMaybeUndef::Scalar(s)
    }
}

impl ScalarMaybeUndef {
    pub fn unwrap_or_err(self) -> EvalResult<'static, Scalar> {
        match self {
            ScalarMaybeUndef::Scalar(scalar) => Ok(scalar),
            ScalarMaybeUndef::Undef => err!(ReadUndefBytes),
        }
    }

    pub fn to_value_with_len<C: HasDataLayout>(self, len: u64, cx: C) -> Value {
        Value::ScalarPair(self.into(), Scalar::Bits {
            bits: len as u128,
            size: cx.data_layout().pointer_size.bytes() as u8,
        }.into())
    }

    pub fn to_value_with_vtable(self, vtable: Pointer) -> Value {
        Value::ScalarPair(self.into(), Scalar::Ptr(vtable).into())
    }

    pub fn ptr_offset<C: HasDataLayout>(self, i: Size, cx: C) -> EvalResult<'tcx, Self> {
        match self {
            ScalarMaybeUndef::Scalar(scalar) => {
                scalar.ptr_offset(i, cx).map(ScalarMaybeUndef::Scalar)
            },
            ScalarMaybeUndef::Undef => Ok(ScalarMaybeUndef::Undef)
        }
    }
}

impl<'tcx> Scalar {
    pub fn from_bool(b: bool) -> Self {
        Scalar::Bits { bits: b as u128, size: 1 }
    }

    pub fn from_char(c: char) -> Self {
        Scalar::Bits { bits: c as u128, size: 4 }
    }

    pub fn to_bits(self, target_size: Size) -> EvalResult<'tcx, u128> {
        match self {
            Scalar::Bits { bits, size } => {
                assert_eq!(target_size.bytes(), size as u64);
                assert_ne!(size, 0, "to_bits cannot be used with zsts");
                Ok(bits)
            }
            Scalar::Ptr(_) => err!(ReadPointerAsBytes),
        }
    }

    pub fn to_ptr(self) -> EvalResult<'tcx, Pointer> {
        match self {
            Scalar::Bits {..} => err!(ReadBytesAsPointer),
            Scalar::Ptr(p) => Ok(p),
        }
    }

    pub fn is_bits(self) -> bool {
        match self {
            Scalar::Bits { .. } => true,
            _ => false,
        }
    }

    pub fn is_ptr(self) -> bool {
        match self {
            Scalar::Ptr(_) => true,
            _ => false,
        }
    }

    pub fn to_bool(self) -> EvalResult<'tcx, bool> {
        match self {
            Scalar::Bits { bits: 0, size: 1 } => Ok(false),
            Scalar::Bits { bits: 1, size: 1 } => Ok(true),
            _ => err!(InvalidBool),
        }
    }
}
