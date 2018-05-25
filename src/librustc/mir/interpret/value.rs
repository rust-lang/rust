#![allow(unknown_lints)]

use ty::layout::{Align, HasDataLayout, Size};
use ty;

use super::{EvalResult, Pointer, PointerArithmetic, Allocation};

/// Represents a constant value in Rust. ByVal and ScalarPair are optimizations which
/// matches Value's optimizations for easy conversions between these two types
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash)]
pub enum ConstValue<'tcx> {
    /// Used only for types with layout::abi::Scalar ABI and ZSTs which use Scalar::undef()
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
    pub fn from_scalar(val: Scalar) -> Self {
        ConstValue::Scalar(val)
    }

    #[inline]
    pub fn to_scalar(&self) -> Option<Scalar> {
        match *self {
            ConstValue::ByRef(..) => None,
            ConstValue::ScalarPair(..) => None,
            ConstValue::Scalar(val) => Some(val),
        }
    }

    #[inline]
    pub fn to_bits(&self, size: Size) -> Option<u128> {
        self.to_scalar()?.to_bits(size).ok()
    }

    #[inline]
    pub fn to_ptr(&self) -> Option<Pointer> {
        self.to_scalar()?.to_ptr().ok()
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
    ByRef(Scalar, Align),
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

impl<'tcx> Scalar {
    pub fn ptr_null<C: HasDataLayout>(cx: C) -> Self {
        Scalar::Bits {
            bits: 0,
            defined: cx.data_layout().pointer_size.bits() as u8,
        }
    }

    pub fn ptr_signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> EvalResult<'tcx, Self> {
        let layout = cx.data_layout();
        match self {
            Scalar::Bits { bits, defined } => {
                let pointer_size = layout.pointer_size.bits() as u8;
                if defined < pointer_size {
                    err!(ReadUndefBytes)
                } else {
                    Ok(Scalar::Bits {
                        bits: layout.signed_offset(bits as u64, i)? as u128,
                        defined: pointer_size,
                    })
            }
            }
            Scalar::Ptr(ptr) => ptr.signed_offset(i, layout).map(Scalar::Ptr),
        }
    }

    pub fn ptr_offset<C: HasDataLayout>(self, i: Size, cx: C) -> EvalResult<'tcx, Self> {
        let layout = cx.data_layout();
        match self {
            Scalar::Bits { bits, defined } => {
                let pointer_size = layout.pointer_size.bits() as u8;
                if defined < pointer_size {
                    err!(ReadUndefBytes)
                } else {
                    Ok(Scalar::Bits {
                        bits: layout.offset(bits as u64, i.bytes())? as u128,
                        defined: pointer_size,
                    })
            }
            }
            Scalar::Ptr(ptr) => ptr.offset(i, layout).map(Scalar::Ptr),
        }
    }

    pub fn ptr_wrapping_signed_offset<C: HasDataLayout>(self, i: i64, cx: C) -> EvalResult<'tcx, Self> {
        let layout = cx.data_layout();
        match self {
            Scalar::Bits { bits, defined } => {
                let pointer_size = layout.pointer_size.bits() as u8;
                if defined < pointer_size {
                    err!(ReadUndefBytes)
                } else {
                    Ok(Scalar::Bits {
                        bits: layout.wrapping_signed_offset(bits as u64, i) as u128,
                        defined: pointer_size,
                    })
            }
            }
            Scalar::Ptr(ptr) => Ok(Scalar::Ptr(ptr.wrapping_signed_offset(i, layout))),
        }
    }

    pub fn is_null_ptr<C: HasDataLayout>(self, cx: C) -> EvalResult<'tcx, bool> {
        match self {
            Scalar::Bits {
                bits, defined,
            } => if defined < cx.data_layout().pointer_size.bits() as u8 {
                err!(ReadUndefBytes)
            } else {
                Ok(bits == 0)
            },
            Scalar::Ptr(_) => Ok(false),
        }
    }

    pub fn to_value_with_len<C: HasDataLayout>(self, len: u64, cx: C) -> Value {
        Value::ScalarPair(self, Scalar::Bits {
            bits: len as u128,
            defined: cx.data_layout().pointer_size.bits() as u8,
        })
    }

    pub fn to_value_with_vtable(self, vtable: Pointer) -> Value {
        Value::ScalarPair(self, Scalar::Ptr(vtable))
    }

    pub fn to_value(self) -> Value {
        Value::Scalar(self)
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
        /// The first `defined` number of bits are valid
        defined: u8,
        bits: u128,
    },

    /// A pointer into an `Allocation`. An `Allocation` in the `memory` module has a list of
    /// relocations, but a `Scalar` is only large enough to contain one, so we just represent the
    /// relocation and its associated offset together as a `Pointer` here.
    Ptr(Pointer),
}

impl<'tcx> Scalar {
    pub fn undef() -> Self {
        Scalar::Bits { bits: 0, defined: 0 }
    }

    pub fn from_bool(b: bool) -> Self {
        // FIXME: can we make defined `1`?
        Scalar::Bits { bits: b as u128, defined: 8 }
    }

    pub fn from_char(c: char) -> Self {
        Scalar::Bits { bits: c as u128, defined: 32 }
    }

    pub fn to_bits(self, size: Size) -> EvalResult<'tcx, u128> {
        match self {
            Scalar::Bits { .. } if size.bits() == 0 => bug!("to_bits cannot be used with zsts"),
            Scalar::Bits { bits, defined } if size.bits() <= defined as u64 => Ok(bits),
            Scalar::Bits { .. } => err!(ReadUndefBytes),
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
            Scalar::Bits { bits: 0, defined: 8 } => Ok(false),
            Scalar::Bits { bits: 1, defined: 8 } => Ok(true),
            _ => err!(InvalidBool),
        }
    }
}
