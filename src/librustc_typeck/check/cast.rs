// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code for type-checking cast expressions.
//!
//! A cast `e as U` is valid if one of the following holds:
//! * `e` has type `T` and `T` coerces to `U`; *coercion-cast*
//! * `e` has type `*T`, `U` is `*U_0`, and either `U_0: Sized` or
//!    unsize_kind(`T`) = unsize_kind(`U_0`); *ptr-ptr-cast*
//! * `e` has type `*T` and `U` is a numeric type, while `T: Sized`; *ptr-addr-cast*
//! * `e` is an integer and `U` is `*U_0`, while `U_0: Sized`; *addr-ptr-cast*
//! * `e` has type `T` and `T` and `U` are any numeric types; *numeric-cast*
//! * `e` is a C-like enum and `U` is an integer type; *enum-cast*
//! * `e` has type `bool` or `char` and `U` is an integer; *prim-int-cast*
//! * `e` has type `u8` and `U` is `char`; *u8-char-cast*
//! * `e` has type `&[T; n]` and `U` is `*const T`; *array-ptr-cast*
//! * `e` is a function pointer type and `U` has type `*T`,
//!   while `T: Sized`; *fptr-ptr-cast*
//! * `e` is a function pointer type and `U` is an integer; *fptr-addr-cast*
//!
//! where `&.T` and `*T` are references of either mutability,
//! and where unsize_kind(`T`) is the kind of the unsize info
//! in `T` - a vtable or a length (or `()` if `T: Sized`).
//!
//! Casting is not transitive, that is, even if `e as U1 as U2` is a valid
//! expression, `e as U2` is not necessarily so (in fact it will only be valid if
//! `U1` coerces to `U2`).

use super::coercion;
use super::demand;
use super::FnCtxt;
use super::structurally_resolved_type;

use lint;
use middle::cast::{CastKind, CastTy};
use middle::ty;
use middle::ty::Ty;
use syntax::ast;
use syntax::ast::UintTy::{TyU8};
use syntax::codemap::Span;
use util::ppaux::Repr;

/// Reifies a cast check to be checked once we have full type information for
/// a function context.
pub struct CastCheck<'tcx> {
    expr: ast::Expr,
    expr_ty: Ty<'tcx>,
    cast_ty: Ty<'tcx>,
    span: Span,
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum UnsizeKind<'tcx> {
    Vtable,
    Length,
    OfTy(Ty<'tcx>)
}

/// Returns the kind of unsize information of t, or None
/// if t is sized or it is unknown.
fn unsize_kind<'a,'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                        t: Ty<'tcx>)
                        -> Option<UnsizeKind<'tcx>> {
    match t.sty {
        ty::ty_vec(_, None) | ty::ty_str => Some(UnsizeKind::Length),
        ty::ty_trait(_) => Some(UnsizeKind::Vtable),
        ty::ty_struct(did, substs) => {
            match ty::struct_fields(fcx.tcx(), did, substs).pop() {
                None => None,
                Some(f) => unsize_kind(fcx, f.mt.ty)
            }
        }
        ty::ty_projection(..) | ty::ty_param(..) =>
            Some(UnsizeKind::OfTy(t)),
        _ => None
    }
}

#[derive(Copy, Clone)]
enum CastError {
    CastToBool,
    CastToChar,
    DifferingKinds,
    IllegalCast,
    NeedViaPtr,
    NeedViaInt,
    NeedViaUsize,
    NonScalar,
    RefToMutPtr
}

impl<'tcx> CastCheck<'tcx> {
    pub fn new(expr: ast::Expr, expr_ty: Ty<'tcx>, cast_ty: Ty<'tcx>, span: Span)
               -> CastCheck<'tcx> {
        CastCheck {
            expr: expr,
            expr_ty: expr_ty,
            cast_ty: cast_ty,
            span: span,
        }
    }

    fn report_cast_error<'a>(&self, fcx: &FnCtxt<'a, 'tcx>,
                             e: CastError) {
        match e {
            CastError::NeedViaPtr |
            CastError::NeedViaInt |
            CastError::NeedViaUsize => {
                fcx.type_error_message(self.span, |actual| {
                    format!("illegal cast; cast through {} first: `{}` as `{}`",
                            match e {
                                CastError::NeedViaPtr => "a raw pointer",
                                CastError::NeedViaInt => "an integer",
                                CastError::NeedViaUsize => "a usize",
                                _ => unreachable!()
                            },
                            actual,
                            fcx.infcx().ty_to_string(self.cast_ty))
                }, self.expr_ty, None)
            }
            CastError::CastToBool => {
                span_err!(fcx.tcx().sess, self.span, E0054,
                          "cannot cast as `bool`, compare with zero instead");
            }
            CastError::CastToChar => {
                fcx.type_error_message(self.span, |actual| {
                    format!("only `u8` can be cast as `char`, not `{}`", actual)
                }, self.expr_ty, None);
            }
            CastError::NonScalar => {
                fcx.type_error_message(self.span, |actual| {
                    format!("non-scalar cast: `{}` as `{}`",
                            actual,
                            fcx.infcx().ty_to_string(self.cast_ty))
                }, self.expr_ty, None);
            }
            CastError::IllegalCast => {
                fcx.type_error_message(self.span, |actual| {
                    format!("illegal cast: `{}` as `{}`",
                            actual,
                            fcx.infcx().ty_to_string(self.cast_ty))
                }, self.expr_ty, None);
            }
            CastError::DifferingKinds => {
                fcx.type_error_message(self.span, |actual| {
                    format!("illegal cast: `{}` as `{}`; vtable kinds may not match",
                            actual,
                            fcx.infcx().ty_to_string(self.cast_ty))
                }, self.expr_ty, None);
            }
            CastError::RefToMutPtr => {
                span_err!(fcx.tcx().sess, self.span, E0188,
                          "cannot cast an immutable reference to a \
                           mutable pointer");
            }
        }
    }

    fn trivial_cast_lint<'a>(&self, fcx: &FnCtxt<'a, 'tcx>) {
        let t_1 = self.cast_ty;
        let t_e = self.expr_ty;
        if ty::type_is_numeric(t_1) && ty::type_is_numeric(t_e) {
            fcx.tcx().sess.add_lint(lint::builtin::TRIVIAL_NUMERIC_CASTS,
                                    self.expr.id,
                                    self.span,
                                    format!("trivial numeric cast: `{}` as `{}`. Cast can be \
                                             replaced by coercion, this might require type \
                                             ascription or a temporary variable",
                                            fcx.infcx().ty_to_string(t_e),
                                            fcx.infcx().ty_to_string(t_1)));
        } else {
            fcx.tcx().sess.add_lint(lint::builtin::TRIVIAL_CASTS,
                                    self.expr.id,
                                    self.span,
                                    format!("trivial cast: `{}` as `{}`. Cast can be \
                                             replaced by coercion, this might require type \
                                             ascription or a temporary variable",
                                            fcx.infcx().ty_to_string(t_e),
                                            fcx.infcx().ty_to_string(t_1)));
        }

    }

    pub fn check<'a>(mut self, fcx: &FnCtxt<'a, 'tcx>) {
        self.expr_ty = structurally_resolved_type(fcx, self.span, self.expr_ty);
        self.cast_ty = structurally_resolved_type(fcx, self.span, self.cast_ty);

        debug!("check_cast({}, {} as {})", self.expr.id, self.expr_ty.repr(fcx.tcx()),
               self.cast_ty.repr(fcx.tcx()));

        if ty::type_is_error(self.expr_ty) || ty::type_is_error(self.cast_ty) {
            // No sense in giving duplicate error messages
        } else if self.try_coercion_cast(fcx) {
            self.trivial_cast_lint(fcx);
            debug!(" -> CoercionCast");
            fcx.tcx().cast_kinds.borrow_mut().insert(self.expr.id,
                                                     CastKind::CoercionCast);
        } else { match self.do_check(fcx) {
            Ok(k) => {
                debug!(" -> {:?}", k);
                fcx.tcx().cast_kinds.borrow_mut().insert(self.expr.id, k);
            }
            Err(e) => self.report_cast_error(fcx, e)
        };}
    }

    /// Check a cast, and report an error if one exists. In some cases,
    /// this can return Ok and create type errors rather than returning
    /// directly. coercion-cast is handled in check instead of here.
    fn do_check<'a>(&self, fcx: &FnCtxt<'a, 'tcx>) -> Result<CastKind, CastError> {
        use middle::cast::IntTy::*;
        use middle::cast::CastTy::*;

        let (t_e, t_1) = match (CastTy::recognize(fcx.tcx(), self.expr_ty),
                                CastTy::recognize(fcx.tcx(), self.cast_ty)) {
            (Some(t_e), Some(t_1)) => (t_e, t_1),
            _ => {
                return Err(CastError::NonScalar)
            }
        };

        match (t_e, t_1) {
            // These types have invariants! can't cast into them.
            (_, RPtr(_)) | (_, Int(CEnum)) | (_, FPtr) => Err(CastError::NonScalar),

            // * -> Bool
            (_, Int(Bool)) => Err(CastError::CastToBool),

            // * -> Char
            (Int(U(ast::TyU8)), Int(Char)) => Ok(CastKind::U8CharCast), // u8-char-cast
            (_, Int(Char)) => Err(CastError::CastToChar),

            // prim -> float,ptr
            (Int(Bool), Float) | (Int(CEnum), Float) | (Int(Char), Float)
                => Err(CastError::NeedViaInt),
            (Int(Bool), Ptr(_)) | (Int(CEnum), Ptr(_)) | (Int(Char), Ptr(_))
                => Err(CastError::NeedViaUsize),

            // ptr -> *
            (Ptr(m1), Ptr(m2)) => self.check_ptr_ptr_cast(fcx, m1, m2), // ptr-ptr-cast
            (Ptr(m_e), Int(_)) => self.check_ptr_addr_cast(fcx, m_e), // ptr-addr-cast
            (Ptr(_), Float) | (FPtr, Float) => Err(CastError::NeedViaUsize),
            (FPtr, Int(_)) => Ok(CastKind::FPtrAddrCast),
            (RPtr(_), Int(_)) | (RPtr(_), Float) => Err(CastError::NeedViaPtr),
            // * -> ptr
            (Int(_), Ptr(mt)) => self.check_addr_ptr_cast(fcx, mt), // addr-ptr-cast
            (FPtr, Ptr(mt)) => self.check_fptr_ptr_cast(fcx, mt),
            (Float, Ptr(_)) => Err(CastError::NeedViaUsize),
            (RPtr(rmt), Ptr(mt)) => self.check_ref_cast(fcx, rmt, mt), // array-ptr-cast

            // prim -> prim
            (Int(CEnum), Int(_)) => Ok(CastKind::EnumCast),
            (Int(Char), Int(_)) | (Int(Bool), Int(_)) => Ok(CastKind::PrimIntCast),

            (Int(_), Int(_)) |
            (Int(_), Float) |
            (Float, Int(_)) |
            (Float, Float) => Ok(CastKind::NumericCast),

        }
    }

    fn check_ptr_ptr_cast<'a>(&self,
                              fcx: &FnCtxt<'a, 'tcx>,
                              m_e: &'tcx ty::mt<'tcx>,
                              m_1: &'tcx ty::mt<'tcx>)
                              -> Result<CastKind, CastError>
    {
        debug!("check_ptr_ptr_cast m_e={} m_1={}",
               m_e.repr(fcx.tcx()), m_1.repr(fcx.tcx()));
        // ptr-ptr cast. vtables must match.

        // Cast to sized is OK
        if fcx.type_is_known_to_be_sized(m_1.ty, self.span) {
            return Ok(CastKind::PtrPtrCast);
        }

        // sized -> unsized? report illegal cast (don't complain about vtable kinds)
        if fcx.type_is_known_to_be_sized(m_e.ty, self.span) {
            return Err(CastError::IllegalCast);
        }

        // vtable kinds must match
        match (unsize_kind(fcx, m_1.ty), unsize_kind(fcx, m_e.ty)) {
            (Some(a), Some(b)) if a == b => Ok(CastKind::PtrPtrCast),
            _ => Err(CastError::DifferingKinds)
        }
    }

    fn check_fptr_ptr_cast<'a>(&self,
                               fcx: &FnCtxt<'a, 'tcx>,
                               m_1: &'tcx ty::mt<'tcx>)
                               -> Result<CastKind, CastError>
    {
        // fptr-ptr cast. must be to sized ptr

        if fcx.type_is_known_to_be_sized(m_1.ty, self.span) {
            Ok(CastKind::FPtrPtrCast)
        } else {
            Err(CastError::IllegalCast)
        }
    }

    fn check_ptr_addr_cast<'a>(&self,
                               fcx: &FnCtxt<'a, 'tcx>,
                               m_e: &'tcx ty::mt<'tcx>)
                               -> Result<CastKind, CastError>
    {
        // ptr-addr cast. must be from sized ptr

        if fcx.type_is_known_to_be_sized(m_e.ty, self.span) {
            Ok(CastKind::PtrAddrCast)
        } else {
            Err(CastError::NeedViaPtr)
        }
    }

    fn check_ref_cast<'a>(&self,
                          fcx: &FnCtxt<'a, 'tcx>,
                          m_e: &'tcx ty::mt<'tcx>,
                          m_1: &'tcx ty::mt<'tcx>)
                          -> Result<CastKind, CastError>
    {
        // array-ptr-cast.

        if m_e.mutbl == ast::MutImmutable && m_1.mutbl == ast::MutImmutable {
            if let ty::ty_vec(ety, Some(_)) = m_e.ty.sty {
                // Due to the limitations of LLVM global constants,
                // region pointers end up pointing at copies of
                // vector elements instead of the original values.
                // To allow unsafe pointers to work correctly, we
                // need to special-case obtaining an unsafe pointer
                // from a region pointer to a vector.
                // TODO: explain comment.

                 // this will report a type mismatch if needed
                demand::eqtype(fcx, self.span, ety, m_1.ty);
                return Ok(CastKind::ArrayPtrCast);
            }
        }

        Err(CastError::IllegalCast)
    }

    fn check_addr_ptr_cast<'a>(&self,
                               fcx: &FnCtxt<'a, 'tcx>,
                               m_1: &'tcx ty::mt<'tcx>)
                               -> Result<CastKind, CastError>
    {
        // ptr-addr cast. pointer must be thin.
        if fcx.type_is_known_to_be_sized(m_1.ty, self.span) {
           Ok(CastKind::AddrPtrCast)
        } else {
           Err(CastError::IllegalCast)
        }
    }

    fn try_coercion_cast<'a>(&self, fcx: &FnCtxt<'a, 'tcx>) -> bool {
        if let Ok(()) = coercion::mk_assignty(fcx,
                                              &self.expr,
                                              self.expr_ty,
                                              self.cast_ty) {
            true
        } else {
            false
        }
    }

}
