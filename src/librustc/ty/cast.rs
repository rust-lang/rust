// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Helpers for handling cast expressions, used in both
// typeck and trans.

use ty::{self, Ty};

use syntax::ast;

/// Types that are represented as ints.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IntTy {
    U(ast::UintTy),
    I,
    CEnum,
    Bool,
    Char
}

// Valid types for the result of a non-coercion cast
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CastTy<'tcx> {
    /// Various types that are represented as ints and handled mostly
    /// in the same way, merged for easier matching.
    Int(IntTy),
    /// Floating-Point types
    Float,
    /// Function Pointers
    FnPtr,
    /// Raw pointers
    Ptr(&'tcx ty::TypeAndMut<'tcx>),
    /// References
    RPtr(&'tcx ty::TypeAndMut<'tcx>),
}

/// Cast Kind. See RFC 401 (or librustc_typeck/check/cast.rs)
#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable)]
pub enum CastKind {
    CoercionCast,
    PtrPtrCast,
    PtrAddrCast,
    AddrPtrCast,
    NumericCast,
    EnumCast,
    PrimIntCast,
    U8CharCast,
    ArrayPtrCast,
    FnPtrPtrCast,
    FnPtrAddrCast
}

impl<'tcx> CastTy<'tcx> {
    pub fn from_ty(t: Ty<'tcx>) -> Option<CastTy<'tcx>> {
        match t.sty {
            ty::TyBool => Some(CastTy::Int(IntTy::Bool)),
            ty::TyChar => Some(CastTy::Int(IntTy::Char)),
            ty::TyInt(_) => Some(CastTy::Int(IntTy::I)),
            ty::TyUint(u) => Some(CastTy::Int(IntTy::U(u))),
            ty::TyFloat(_) => Some(CastTy::Float),
            ty::TyAdt(d,_) if d.is_enum() && d.is_payloadfree() =>
                Some(CastTy::Int(IntTy::CEnum)),
            ty::TyRawPtr(ref mt) => Some(CastTy::Ptr(mt)),
            ty::TyRef(_, ref mt) => Some(CastTy::RPtr(mt)),
            ty::TyFnPtr(..) => Some(CastTy::FnPtr),
            _ => None,
        }
    }
}
