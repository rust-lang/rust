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
//! in `T` - the vtable for a trait definition (e.g. `fmt::Display` or
//! `Iterator`, not `Iterator<Item=u8>`) or a length (or `()` if `T: Sized`).
//!
//! Note that lengths are not adjusted when casting raw slices -
//! `T: *const [u16] as *const [u8]` creates a slice that only includes
//! half of the original memory.
//!
//! Casting is not transitive, that is, even if `e as U1 as U2` is a valid
//! expression, `e as U2` is not necessarily so (in fact it will only be valid if
//! `U1` coerces to `U2`).

use super::coercion;
use super::demand;
use super::FnCtxt;
use super::structurally_resolved_type;

use lint;
use middle::def_id::DefId;
use middle::ty::{self, Ty, HasTypeFlags};
use middle::ty::cast::{CastKind, CastTy};
use syntax::codemap::Span;
use rustc_front::hir;
use syntax::ast;
use syntax::ast::UintTy::TyU8;


/// Reifies a cast check to be checked once we have full type information for
/// a function context.
pub struct CastCheck<'tcx> {
    expr: hir::Expr,
    expr_ty: Ty<'tcx>,
    cast_ty: Ty<'tcx>,
    span: Span,
}

/// The kind of the unsize info (length or vtable) - we only allow casts between
/// fat pointers if their unsize-infos have the same kind.
#[derive(Copy, Clone, PartialEq, Eq)]
enum UnsizeKind<'tcx> {
    Vtable(DefId),
    Length,
    /// The unsize info of this projection
    OfProjection(&'tcx ty::ProjectionTy<'tcx>),
    /// The unsize info of this parameter
    OfParam(&'tcx ty::ParamTy)
}

/// Returns the kind of unsize information of t, or None
/// if t is sized or it is unknown.
fn unsize_kind<'a,'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                        t: Ty<'tcx>)
                        -> Option<UnsizeKind<'tcx>> {
    match t.sty {
        ty::TySlice(_) | ty::TyStr => Some(UnsizeKind::Length),
        ty::TyTrait(ref tty) => Some(UnsizeKind::Vtable(tty.principal_def_id())),
        ty::TyStruct(def, substs) => {
            // FIXME(arielb1): do some kind of normalization
            match def.struct_variant().fields.last() {
                None => None,
                Some(f) => unsize_kind(fcx, f.ty(fcx.tcx(), substs))
            }
        }
        // We should really try to normalize here.
        ty::TyProjection(ref pi) => Some(UnsizeKind::OfProjection(pi)),
        ty::TyParam(ref p) => Some(UnsizeKind::OfParam(p)),
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
    NeedViaThinPtr,
    NeedViaInt,
    NeedViaUsize,
    NonScalar,
}

impl<'tcx> CastCheck<'tcx> {
    pub fn new(expr: hir::Expr, expr_ty: Ty<'tcx>, cast_ty: Ty<'tcx>, span: Span)
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
            CastError::NeedViaThinPtr |
            CastError::NeedViaInt |
            CastError::NeedViaUsize => {
                fcx.type_error_struct(self.span, |actual| {
                    format!("casting `{}` as `{}` is invalid",
                            actual,
                            fcx.infcx().ty_to_string(self.cast_ty))
                }, self.expr_ty, None)
                    .fileline_help(self.span,
                        &format!("cast through {} first", match e {
                            CastError::NeedViaPtr => "a raw pointer",
                            CastError::NeedViaThinPtr => "a thin pointer",
                            CastError::NeedViaInt => "an integer",
                            CastError::NeedViaUsize => "a usize",
                            _ => unreachable!()
                        }))
                    .emit();
            }
            CastError::CastToBool => {
                struct_span_err!(fcx.tcx().sess, self.span, E0054, "cannot cast as `bool`")
                    .fileline_help(self.span, "compare with zero instead")
                    .emit();
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
                    format!("casting `{}` as `{}` is invalid",
                            actual,
                            fcx.infcx().ty_to_string(self.cast_ty))
                }, self.expr_ty, None);
            }
            CastError::DifferingKinds => {
                fcx.type_error_struct(self.span, |actual| {
                    format!("casting `{}` as `{}` is invalid",
                            actual,
                            fcx.infcx().ty_to_string(self.cast_ty))
                }, self.expr_ty, None)
                    .fileline_note(self.span, "vtable kinds may not match")
                    .emit();
            }
        }
    }

    fn trivial_cast_lint<'a>(&self, fcx: &FnCtxt<'a, 'tcx>) {
        let t_cast = self.cast_ty;
        let t_expr = self.expr_ty;
        if t_cast.is_numeric() && t_expr.is_numeric() {
            fcx.tcx().sess.add_lint(lint::builtin::TRIVIAL_NUMERIC_CASTS,
                                    self.expr.id,
                                    self.span,
                                    format!("trivial numeric cast: `{}` as `{}`. Cast can be \
                                             replaced by coercion, this might require type \
                                             ascription or a temporary variable",
                                            fcx.infcx().ty_to_string(t_expr),
                                            fcx.infcx().ty_to_string(t_cast)));
        } else {
            fcx.tcx().sess.add_lint(lint::builtin::TRIVIAL_CASTS,
                                    self.expr.id,
                                    self.span,
                                    format!("trivial cast: `{}` as `{}`. Cast can be \
                                             replaced by coercion, this might require type \
                                             ascription or a temporary variable",
                                            fcx.infcx().ty_to_string(t_expr),
                                            fcx.infcx().ty_to_string(t_cast)));
        }

    }

    pub fn check<'a>(mut self, fcx: &FnCtxt<'a, 'tcx>) {
        self.expr_ty = structurally_resolved_type(fcx, self.span, self.expr_ty);
        self.cast_ty = structurally_resolved_type(fcx, self.span, self.cast_ty);

        debug!("check_cast({}, {:?} as {:?})", self.expr.id, self.expr_ty,
               self.cast_ty);

        if self.expr_ty.references_error() || self.cast_ty.references_error() {
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

    /// Check a cast, and report an error if one exists. In some cases, this
    /// can return Ok and create type errors in the fcx rather than returning
    /// directly. coercion-cast is handled in check instead of here.
    fn do_check<'a>(&self, fcx: &FnCtxt<'a, 'tcx>) -> Result<CastKind, CastError> {
        use middle::ty::cast::IntTy::*;
        use middle::ty::cast::CastTy::*;

        let (t_from, t_cast) = match (CastTy::from_ty(self.expr_ty),
                                      CastTy::from_ty(self.cast_ty)) {
            (Some(t_from), Some(t_cast)) => (t_from, t_cast),
            _ => {
                return Err(CastError::NonScalar)
            }
        };

        match (t_from, t_cast) {
            // These types have invariants! can't cast into them.
            (_, RPtr(_)) | (_, Int(CEnum)) | (_, FnPtr) => Err(CastError::NonScalar),

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
            (Ptr(m_e), Ptr(m_c)) => self.check_ptr_ptr_cast(fcx, m_e, m_c), // ptr-ptr-cast
            (Ptr(m_expr), Int(_)) => self.check_ptr_addr_cast(fcx, m_expr), // ptr-addr-cast
            (Ptr(_), Float) | (FnPtr, Float) => Err(CastError::NeedViaUsize),
            (FnPtr, Int(_)) => Ok(CastKind::FnPtrAddrCast),
            (RPtr(_), Int(_)) | (RPtr(_), Float) => Err(CastError::NeedViaPtr),
            // * -> ptr
            (Int(_), Ptr(mt)) => self.check_addr_ptr_cast(fcx, mt), // addr-ptr-cast
            (FnPtr, Ptr(mt)) => self.check_fptr_ptr_cast(fcx, mt),
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
                              m_expr: &'tcx ty::TypeAndMut<'tcx>,
                              m_cast: &'tcx ty::TypeAndMut<'tcx>)
                              -> Result<CastKind, CastError>
    {
        debug!("check_ptr_ptr_cast m_expr={:?} m_cast={:?}",
               m_expr, m_cast);
        // ptr-ptr cast. vtables must match.

        // Cast to sized is OK
        if fcx.type_is_known_to_be_sized(m_cast.ty, self.span) {
            return Ok(CastKind::PtrPtrCast);
        }

        // sized -> unsized? report invalid cast (don't complain about vtable kinds)
        if fcx.type_is_known_to_be_sized(m_expr.ty, self.span) {
            return Err(CastError::IllegalCast);
        }

        // vtable kinds must match
        match (unsize_kind(fcx, m_cast.ty), unsize_kind(fcx, m_expr.ty)) {
            (Some(a), Some(b)) if a == b => Ok(CastKind::PtrPtrCast),
            _ => Err(CastError::DifferingKinds)
        }
    }

    fn check_fptr_ptr_cast<'a>(&self,
                               fcx: &FnCtxt<'a, 'tcx>,
                               m_cast: &'tcx ty::TypeAndMut<'tcx>)
                               -> Result<CastKind, CastError>
    {
        // fptr-ptr cast. must be to sized ptr

        if fcx.type_is_known_to_be_sized(m_cast.ty, self.span) {
            Ok(CastKind::FnPtrPtrCast)
        } else {
            Err(CastError::IllegalCast)
        }
    }

    fn check_ptr_addr_cast<'a>(&self,
                               fcx: &FnCtxt<'a, 'tcx>,
                               m_expr: &'tcx ty::TypeAndMut<'tcx>)
                               -> Result<CastKind, CastError>
    {
        // ptr-addr cast. must be from sized ptr

        if fcx.type_is_known_to_be_sized(m_expr.ty, self.span) {
            Ok(CastKind::PtrAddrCast)
        } else {
            Err(CastError::NeedViaThinPtr)
        }
    }

    fn check_ref_cast<'a>(&self,
                          fcx: &FnCtxt<'a, 'tcx>,
                          m_expr: &'tcx ty::TypeAndMut<'tcx>,
                          m_cast: &'tcx ty::TypeAndMut<'tcx>)
                          -> Result<CastKind, CastError>
    {
        // array-ptr-cast.

        if m_expr.mutbl == hir::MutImmutable && m_cast.mutbl == hir::MutImmutable {
            if let ty::TyArray(ety, _) = m_expr.ty.sty {
                // Due to the limitations of LLVM global constants,
                // region pointers end up pointing at copies of
                // vector elements instead of the original values.
                // To allow raw pointers to work correctly, we
                // need to special-case obtaining a raw pointer
                // from a region pointer to a vector.

                // this will report a type mismatch if needed
                demand::eqtype(fcx, self.span, ety, m_cast.ty);
                return Ok(CastKind::ArrayPtrCast);
            }
        }

        Err(CastError::IllegalCast)
    }

    fn check_addr_ptr_cast<'a>(&self,
                               fcx: &FnCtxt<'a, 'tcx>,
                               m_cast: &'tcx ty::TypeAndMut<'tcx>)
                               -> Result<CastKind, CastError>
    {
        // ptr-addr cast. pointer must be thin.
        if fcx.type_is_known_to_be_sized(m_cast.ty, self.span) {
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
