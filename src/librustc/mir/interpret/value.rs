// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_lints)]

use ty::layout::{HasDataLayout, Size};
use ty::subst::Substs;
use hir::def_id::DefId;

use super::{EvalResult, Pointer, PointerArithmetic, Allocation};

/// Represents a constant value in Rust. Scalar and ScalarPair are optimizations which
/// matches the LocalValue optimizations for easy conversions between Value and ConstValue.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, RustcEncodable, RustcDecodable, Hash)]
pub enum ConstValue<'tcx> {
    /// Never returned from the `const_eval` query, but the HIR contains these frequently in order
    /// to allow HIR creation to happen for everything before needing to be able to run constant
    /// evaluation
    Unevaluated(DefId, &'tcx Substs<'tcx>),
    /// Used only for types with layout::abi::Scalar ABI and ZSTs
    ///
    /// Not using the enum `Value` to encode that this must not be `Undef`
    Scalar(Scalar),
    /// Used only for types with layout::abi::ScalarPair
    ///
    /// The second field may be undef in case of `Option<usize>::None`
    ScalarPair(Scalar, ScalarMaybeUndef),
    /// Used only for the remaining cases. An allocation + offset into the allocation
    ByRef(&'tcx Allocation, Size),
}

impl<'tcx> ConstValue<'tcx> {
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
    pub fn try_to_bits(&self, size: Size) -> Option<u128> {
        self.try_to_scalar()?.to_bits(size).ok()
    }

    #[inline]
    pub fn try_to_ptr(&self) -> Option<Pointer> {
        self.try_to_scalar()?.to_ptr().ok()
    }

    pub fn new_slice(
        val: Scalar,
        len: u64,
        cx: impl HasDataLayout
    ) -> Self {
        ConstValue::ScalarPair(val, Scalar::Bits {
            bits: len as u128,
            size: cx.data_layout().pointer_size.bytes() as u8,
        }.into())
    }

    pub fn new_dyn_trait(val: Scalar, vtable: Pointer) -> Self {
        ConstValue::ScalarPair(val, Scalar::Ptr(vtable).into())
    }
}

impl<'tcx> Scalar {
    pub fn ptr_null(cx: impl HasDataLayout) -> Self {
        Scalar::Bits {
            bits: 0,
            size: cx.data_layout().pointer_size.bytes() as u8,
        }
    }

    pub fn zst() -> Self {
        Scalar::Bits { bits: 0, size: 0 }
    }

    pub fn ptr_signed_offset(self, i: i64, cx: impl HasDataLayout) -> EvalResult<'tcx, Self> {
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

    pub fn ptr_offset(self, i: Size, cx: impl HasDataLayout) -> EvalResult<'tcx, Self> {
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

    pub fn ptr_wrapping_signed_offset(self, i: i64, cx: impl HasDataLayout) -> Self {
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

    pub fn is_null_ptr(self, cx: impl HasDataLayout) -> bool {
        match self {
            Scalar::Bits { bits, size } =>  {
                assert_eq!(size as u64, cx.data_layout().pointer_size.bytes());
                bits == 0
            },
            Scalar::Ptr(_) => false,
        }
    }

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
            Scalar::Bits { bits: 0, .. } => err!(InvalidNullPointerUsage),
            Scalar::Bits { .. } => err!(ReadBytesAsPointer),
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

impl From<Pointer> for Scalar {
    #[inline(always)]
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
        /// Do not try to read less or more bytes that that. The remaining bytes must be 0.
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
    #[inline(always)]
    fn from(s: Scalar) -> Self {
        ScalarMaybeUndef::Scalar(s)
    }
}

impl<'tcx> ScalarMaybeUndef {
    pub fn not_undef(self) -> EvalResult<'static, Scalar> {
        match self {
            ScalarMaybeUndef::Scalar(scalar) => Ok(scalar),
            ScalarMaybeUndef::Undef => err!(ReadUndefBytes),
        }
    }

    pub fn to_ptr(self) -> EvalResult<'tcx, Pointer> {
        self.not_undef()?.to_ptr()
    }

    pub fn to_bits(self, target_size: Size) -> EvalResult<'tcx, u128> {
        self.not_undef()?.to_bits(target_size)
    }

    pub fn to_bool(self) -> EvalResult<'tcx, bool> {
        self.not_undef()?.to_bool()
    }
}
