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

use super::coercion;
use super::demand;
use super::FnCtxt;
use super::structurally_resolved_type;

use lint;
use middle::infer;
use middle::ty;
use middle::ty::Ty;
use syntax::ast;
use syntax::codemap::Span;

/// Reifies a cast check to be checked once we have full type information for
/// a function context.
pub struct CastCheck<'tcx> {
    expr: ast::Expr,
    expr_ty: Ty<'tcx>,
    cast_ty: Ty<'tcx>,
    span: Span,
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
}

pub fn check_cast<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>, cast: &CastCheck<'tcx>) {
    fn cast_through_integer_err<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                          span: Span,
                                          t_1: Ty<'tcx>,
                                          t_e: Ty<'tcx>) {
        fcx.type_error_message(span, |actual| {
            format!("illegal cast; cast through an \
                    integer first: `{}` as `{}`",
                    actual,
                    fcx.infcx().ty_to_string(t_1))
        }, t_e, None);
    }

    let span = cast.span;
    let e = &cast.expr;
    let t_e = structurally_resolved_type(fcx, span, cast.expr_ty);
    let t_1 = structurally_resolved_type(fcx, span, cast.cast_ty);

    // Check for trivial casts.
    if !ty::type_has_ty_infer(t_1) {
        if let Ok(()) = coercion::mk_assignty(fcx, e, t_e, t_1) {
            if ty::type_is_numeric(t_1) && ty::type_is_numeric(t_e) {
                fcx.tcx().sess.add_lint(lint::builtin::TRIVIAL_NUMERIC_CASTS,
                                        e.id,
                                        span,
                                        format!("trivial numeric cast: `{}` as `{}`. Cast can be \
                                                 replaced by coercion, this might require type \
                                                 ascription or a temporary variable",
                                                fcx.infcx().ty_to_string(t_e),
                                                fcx.infcx().ty_to_string(t_1)));
            } else {
                fcx.tcx().sess.add_lint(lint::builtin::TRIVIAL_CASTS,
                                        e.id,
                                        span,
                                        format!("trivial cast: `{}` as `{}`. Cast can be \
                                                 replaced by coercion, this might require type \
                                                 ascription or a temporary variable",
                                                fcx.infcx().ty_to_string(t_e),
                                                fcx.infcx().ty_to_string(t_1)));
            }
            return;
        }
    }

    let t_e_is_bare_fn_item = ty::type_is_bare_fn_item(t_e);
    let t_e_is_scalar = ty::type_is_scalar(t_e);
    let t_e_is_integral = ty::type_is_integral(t_e);
    let t_e_is_float = ty::type_is_floating_point(t_e);
    let t_e_is_c_enum = ty::type_is_c_like_enum(fcx.tcx(), t_e);

    let t_1_is_scalar = ty::type_is_scalar(t_1);
    let t_1_is_integral = ty::type_is_integral(t_1);
    let t_1_is_char = ty::type_is_char(t_1);
    let t_1_is_bare_fn = ty::type_is_bare_fn(t_1);
    let t_1_is_float = ty::type_is_floating_point(t_1);
    let t_1_is_c_enum = ty::type_is_c_like_enum(fcx.tcx(), t_1);

    // casts to scalars other than `char` and `bare fn` are trivial
    let t_1_is_trivial = t_1_is_scalar && !t_1_is_char && !t_1_is_bare_fn;

    if t_e_is_bare_fn_item && t_1_is_bare_fn {
        demand::coerce(fcx, e.span, t_1, &e);
    } else if t_1_is_char {
        let t_e = fcx.infcx().shallow_resolve(t_e);
        if t_e.sty != ty::ty_uint(ast::TyU8) {
            fcx.type_error_message(span, |actual| {
                format!("only `u8` can be cast as `char`, not `{}`", actual)
            }, t_e, None);
        }
    } else if t_1.sty == ty::ty_bool {
        span_err!(fcx.tcx().sess, span, E0054,
                  "cannot cast as `bool`, compare with zero instead");
    } else if t_e_is_float && (t_1_is_scalar || t_1_is_c_enum) &&
        !(t_1_is_integral || t_1_is_float) {
        // Casts from float must go through an integer
        cast_through_integer_err(fcx, span, t_1, t_e)
    } else if t_1_is_float && (t_e_is_scalar || t_e_is_c_enum) &&
        !(t_e_is_integral || t_e_is_float || t_e.sty == ty::ty_bool) {
        // Casts to float must go through an integer or boolean
        cast_through_integer_err(fcx, span, t_1, t_e)
    } else if t_e_is_c_enum && t_1_is_trivial {
        if ty::type_is_unsafe_ptr(t_1) {
            // ... and likewise with C enum -> *T
            cast_through_integer_err(fcx, span, t_1, t_e)
        }
        // casts from C-like enums are allowed
    } else if ty::type_is_region_ptr(t_e) && ty::type_is_unsafe_ptr(t_1) {
        fn types_compatible<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>, sp: Span,
                                      t1: Ty<'tcx>, t2: Ty<'tcx>) -> bool {
            match t1.sty {
                ty::ty_vec(_, Some(_)) => {}
                _ => return false
            }
            if ty::type_needs_infer(t2) {
                // This prevents this special case from going off when casting
                // to a type that isn't fully specified; e.g. `as *_`. (Issue
                // #14893.)
                return false
            }

            let el = ty::sequence_element_type(fcx.tcx(), t1);
            infer::mk_eqty(fcx.infcx(),
                           false,
                           infer::Misc(sp),
                           el,
                           t2).is_ok()
        }

        // Due to the limitations of LLVM global constants,
        // region pointers end up pointing at copies of
        // vector elements instead of the original values.
        // To allow unsafe pointers to work correctly, we
        // need to special-case obtaining an unsafe pointer
        // from a region pointer to a vector.

        /* this cast is only allowed from &[T, ..n] to *T or
        &T to *T. */
        match (&t_e.sty, &t_1.sty) {
            (&ty::ty_rptr(_, ty::mt { ty: mt1, mutbl: ast::MutImmutable }),
             &ty::ty_ptr(ty::mt { ty: mt2, mutbl: ast::MutImmutable }))
            if types_compatible(fcx, e.span, mt1, mt2) => {
                /* this case is allowed */
            }
            _ => {
                demand::coerce(fcx, e.span, t_1, &e);
            }
        }
    } else if fcx.type_is_fat_ptr(t_e, span) && !fcx.type_is_fat_ptr(t_1, span) {
        fcx.type_error_message(span, |actual| {
            format!("illegal cast; cast from fat pointer: `{}` as `{}`",
                    actual, fcx.infcx().ty_to_string(t_1))
        }, t_e, None);
    } else if !(t_e_is_scalar && t_1_is_trivial) {
        /*
        If more type combinations should be supported than are
        supported here, then file an enhancement issue and
        record the issue number in this comment.
        */
        fcx.type_error_message(span, |actual| {
            format!("non-scalar cast: `{}` as `{}`",
                    actual,
                    fcx.infcx().ty_to_string(t_1))
        }, t_e, None);
    }
}
