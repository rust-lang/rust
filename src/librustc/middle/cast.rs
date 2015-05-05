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

use middle::ty::{self, Ty};

use syntax::ast;


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
    Int(IntTy),
    Float,
    FPtr,
    Ptr(&'tcx ty::mt<'tcx>),
    RPtr(&'tcx ty::mt<'tcx>),
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
    FPtrPtrCast,
    FPtrAddrCast
}

impl<'tcx> CastTy<'tcx> {
    pub fn recognize(tcx: &ty::ctxt<'tcx>, t: Ty<'tcx>)
                     -> Option<CastTy<'tcx>> {
        match t.sty {
            ty::ty_bool => Some(CastTy::Int(IntTy::Bool)),
            ty::ty_char => Some(CastTy::Int(IntTy::Char)),
            ty::ty_int(_) => Some(CastTy::Int(IntTy::I)),
            ty::ty_uint(u) => Some(CastTy::Int(IntTy::U(u))),
            ty::ty_float(_) => Some(CastTy::Float),
            ty::ty_enum(..) if ty::type_is_c_like_enum(
                tcx, t) => Some(CastTy::Int(IntTy::CEnum)),
            ty::ty_ptr(ref mt) => Some(CastTy::Ptr(mt)),
            ty::ty_rptr(_, ref mt) => Some(CastTy::RPtr(mt)),
            ty::ty_bare_fn(..) => Some(CastTy::FPtr),
            _ => None,
        }
    }
}
