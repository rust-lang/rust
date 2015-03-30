// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code related to processing overloaded binary and unary operators.

use super::{
    check_expr,
    check_expr_coercable_to_type,
    check_expr_with_lvalue_pref,
    demand,
    method,
    FnCtxt,
    PreferMutLvalue,
    structurally_resolved_type,
};
use middle::infer;
use middle::traits;
use middle::ty::{self, Ty};
use syntax::ast;
use syntax::ast_util;
use syntax::parse::token;
use util::ppaux::{Repr, UserString};

/// Check a `a <op>= b`
pub fn check_binop_assign<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                                   expr: &'tcx ast::Expr,
                                   op: ast::BinOp,
                                   lhs_expr: &'tcx ast::Expr,
                                   rhs_expr: &'tcx ast::Expr)
{
    let tcx = fcx.ccx.tcx;

    check_expr_with_lvalue_pref(fcx, lhs_expr, PreferMutLvalue);
    check_expr(fcx, rhs_expr);

    let lhs_ty = structurally_resolved_type(fcx, lhs_expr.span, fcx.expr_ty(lhs_expr));
    let rhs_ty = structurally_resolved_type(fcx, rhs_expr.span, fcx.expr_ty(rhs_expr));

    if is_builtin_binop(fcx.tcx(), lhs_ty, rhs_ty, op) {
        enforce_builtin_binop_types(fcx, lhs_expr, lhs_ty, rhs_expr, rhs_ty, op);
        fcx.write_nil(expr.id);
    } else {
        // error types are considered "builtin"
        assert!(!ty::type_is_error(lhs_ty) || !ty::type_is_error(rhs_ty));
        span_err!(tcx.sess, lhs_expr.span, E0368,
                  "binary assignment operation `{}=` cannot be applied to types `{}` and `{}`",
                  ast_util::binop_to_string(op.node),
                  lhs_ty.user_string(fcx.tcx()),
                  rhs_ty.user_string(fcx.tcx()));
        fcx.write_error(expr.id);
    }

    let tcx = fcx.tcx();
    if !ty::expr_is_lval(tcx, lhs_expr) {
        span_err!(tcx.sess, lhs_expr.span, E0067, "illegal left-hand side expression");
    }

    fcx.require_expr_have_sized_type(lhs_expr, traits::AssignmentLhsSized);
}

/// Check a potentially overloaded binary operator.
pub fn check_binop<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                             expr: &'tcx ast::Expr,
                             op: ast::BinOp,
                             lhs_expr: &'tcx ast::Expr,
                             rhs_expr: &'tcx ast::Expr)
{
    let tcx = fcx.ccx.tcx;

    debug!("check_binop(expr.id={}, expr={}, op={:?}, lhs_expr={}, rhs_expr={})",
           expr.id,
           expr.repr(tcx),
           op,
           lhs_expr.repr(tcx),
           rhs_expr.repr(tcx));

    check_expr(fcx, lhs_expr);
    let lhs_ty = fcx.resolve_type_vars_if_possible(fcx.expr_ty(lhs_expr));

    // Annoyingly, SIMD ops don't fit into the PartialEq/PartialOrd
    // traits, because their return type is not bool. Perhaps this
    // should change, but for now if LHS is SIMD we go down a
    // different path that bypassess all traits.
    if ty::type_is_simd(fcx.tcx(), lhs_ty) {
        check_expr_coercable_to_type(fcx, rhs_expr, lhs_ty);
        let rhs_ty = fcx.resolve_type_vars_if_possible(fcx.expr_ty(lhs_expr));
        let return_ty = enforce_builtin_binop_types(fcx, lhs_expr, lhs_ty, rhs_expr, rhs_ty, op);
        fcx.write_ty(expr.id, return_ty);
        return;
    }

    match BinOpCategory::from(op) {
        BinOpCategory::Shortcircuit => {
            // && and || are a simple case.
            demand::suptype(fcx, lhs_expr.span, ty::mk_bool(tcx), lhs_ty);
            check_expr_coercable_to_type(fcx, rhs_expr, ty::mk_bool(tcx));
            fcx.write_ty(expr.id, ty::mk_bool(tcx));
        }
        _ => {
            // Otherwise, we always treat operators as if they are
            // overloaded. This is the way to be most flexible w/r/t
            // types that get inferred.
            let (rhs_ty, return_ty) =
                check_overloaded_binop(fcx, expr, lhs_expr, lhs_ty, rhs_expr, op);

            // Supply type inference hints if relevant. Probably these
            // hints should be enforced during select as part of the
            // `consider_unification_despite_ambiguity` routine, but this
            // more convenient for now.
            //
            // The basic idea is to help type inference by taking
            // advantage of things we know about how the impls for
            // scalar types are arranged. This is important in a
            // scenario like `1_u32 << 2`, because it lets us quickly
            // deduce that the result type should be `u32`, even
            // though we don't know yet what type 2 has and hence
            // can't pin this down to a specific impl.
            let rhs_ty = fcx.resolve_type_vars_if_possible(rhs_ty);
            if
                !ty::type_is_ty_var(lhs_ty) &&
                !ty::type_is_ty_var(rhs_ty) &&
                is_builtin_binop(fcx.tcx(), lhs_ty, rhs_ty, op)
            {
                let builtin_return_ty =
                    enforce_builtin_binop_types(fcx, lhs_expr, lhs_ty, rhs_expr, rhs_ty, op);
                demand::suptype(fcx, expr.span, builtin_return_ty, return_ty);
            }

            fcx.write_ty(expr.id, return_ty);
        }
    }
}

fn enforce_builtin_binop_types<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                         lhs_expr: &'tcx ast::Expr,
                                         lhs_ty: Ty<'tcx>,
                                         rhs_expr: &'tcx ast::Expr,
                                         rhs_ty: Ty<'tcx>,
                                         op: ast::BinOp)
                                         -> Ty<'tcx>
{
    debug_assert!(is_builtin_binop(fcx.tcx(), lhs_ty, rhs_ty, op));

    let tcx = fcx.tcx();
    match BinOpCategory::from(op) {
        BinOpCategory::Shortcircuit => {
            demand::suptype(fcx, lhs_expr.span, ty::mk_bool(tcx), lhs_ty);
            demand::suptype(fcx, rhs_expr.span, ty::mk_bool(tcx), rhs_ty);
            ty::mk_bool(tcx)
        }

        BinOpCategory::Shift => {
            // For integers, the shift amount can be of any integral
            // type. For simd, the type must match exactly.
            if ty::type_is_simd(tcx, lhs_ty) {
                demand::suptype(fcx, rhs_expr.span, lhs_ty, rhs_ty);
            }

            // result type is same as LHS always
            lhs_ty
        }

        BinOpCategory::Math |
        BinOpCategory::Bitwise => {
            // both LHS and RHS and result will have the same type
            demand::suptype(fcx, rhs_expr.span, lhs_ty, rhs_ty);
            lhs_ty
        }

        BinOpCategory::Comparison => {
            // both LHS and RHS and result will have the same type
            demand::suptype(fcx, rhs_expr.span, lhs_ty, rhs_ty);

            // if this is simd, result is same as lhs, else bool
            if ty::type_is_simd(tcx, lhs_ty) {
                let unit_ty = ty::simd_type(tcx, lhs_ty);
                debug!("enforce_builtin_binop_types: lhs_ty={} unit_ty={}",
                       lhs_ty.repr(tcx),
                       unit_ty.repr(tcx));
                if !ty::type_is_integral(unit_ty) {
                    tcx.sess.span_err(
                        lhs_expr.span,
                        &format!("binary comparison operation `{}` not supported \
                                  for floating point SIMD vector `{}`",
                                 ast_util::binop_to_string(op.node),
                                 lhs_ty.user_string(tcx)));
                    tcx.types.err
                } else {
                    lhs_ty
                }
            } else {
                ty::mk_bool(tcx)
            }
        }
    }
}

fn check_overloaded_binop<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                    expr: &'tcx ast::Expr,
                                    lhs_expr: &'tcx ast::Expr,
                                    lhs_ty: Ty<'tcx>,
                                    rhs_expr: &'tcx ast::Expr,
                                    op: ast::BinOp)
                                    -> (Ty<'tcx>, Ty<'tcx>)
{
    debug!("check_overloaded_binop(expr.id={}, lhs_ty={})",
           expr.id,
           lhs_ty.repr(fcx.tcx()));

    let (name, trait_def_id) = name_and_trait_def_id(fcx, op);

    // NB: As we have not yet type-checked the RHS, we don't have the
    // type at hand. Make a variable to represent it. The whole reason
    // for this indirection is so that, below, we can check the expr
    // using this variable as the expected type, which sometimes lets
    // us do better coercions than we would be able to do otherwise,
    // particularly for things like `String + &String`.
    let rhs_ty_var = fcx.infcx().next_ty_var();

    let return_ty = match lookup_op_method(fcx, expr, lhs_ty, vec![rhs_ty_var],
                                           token::intern(name), trait_def_id,
                                           lhs_expr) {
        Ok(return_ty) => return_ty,
        Err(()) => {
            // error types are considered "builtin"
            if !ty::type_is_error(lhs_ty) {
                span_err!(fcx.tcx().sess, lhs_expr.span, E0369,
                          "binary operation `{}` cannot be applied to type `{}`",
                          ast_util::binop_to_string(op.node),
                          lhs_ty.user_string(fcx.tcx()));
            }
            fcx.tcx().types.err
        }
    };

    // see `NB` above
    check_expr_coercable_to_type(fcx, rhs_expr, rhs_ty_var);

    (rhs_ty_var, return_ty)
}

pub fn check_user_unop<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                 op_str: &str,
                                 mname: &str,
                                 trait_did: Option<ast::DefId>,
                                 ex: &'tcx ast::Expr,
                                 operand_expr: &'tcx ast::Expr,
                                 operand_ty: Ty<'tcx>,
                                 op: ast::UnOp)
                                 -> Ty<'tcx>
{
    assert!(ast_util::is_by_value_unop(op));
    match lookup_op_method(fcx, ex, operand_ty, vec![],
                           token::intern(mname), trait_did,
                           operand_expr) {
        Ok(t) => t,
        Err(()) => {
            fcx.type_error_message(ex.span, |actual| {
                format!("cannot apply unary operator `{}` to type `{}`",
                        op_str, actual)
            }, operand_ty, None);
            fcx.tcx().types.err
        }
    }
}

fn name_and_trait_def_id(fcx: &FnCtxt, op: ast::BinOp) -> (&'static str, Option<ast::DefId>) {
    let lang = &fcx.tcx().lang_items;
    match op.node {
        ast::BiAdd => ("add", lang.add_trait()),
        ast::BiSub => ("sub", lang.sub_trait()),
        ast::BiMul => ("mul", lang.mul_trait()),
        ast::BiDiv => ("div", lang.div_trait()),
        ast::BiRem => ("rem", lang.rem_trait()),
        ast::BiBitXor => ("bitxor", lang.bitxor_trait()),
        ast::BiBitAnd => ("bitand", lang.bitand_trait()),
        ast::BiBitOr => ("bitor", lang.bitor_trait()),
        ast::BiShl => ("shl", lang.shl_trait()),
        ast::BiShr => ("shr", lang.shr_trait()),
        ast::BiLt => ("lt", lang.ord_trait()),
        ast::BiLe => ("le", lang.ord_trait()),
        ast::BiGe => ("ge", lang.ord_trait()),
        ast::BiGt => ("gt", lang.ord_trait()),
        ast::BiEq => ("eq", lang.eq_trait()),
        ast::BiNe => ("ne", lang.eq_trait()),
        ast::BiAnd | ast::BiOr => {
            fcx.tcx().sess.span_bug(op.span, "&& and || are not overloadable")
        }
    }
}

fn lookup_op_method<'a, 'tcx>(fcx: &'a FnCtxt<'a, 'tcx>,
                              expr: &'tcx ast::Expr,
                              lhs_ty: Ty<'tcx>,
                              other_tys: Vec<Ty<'tcx>>,
                              opname: ast::Name,
                              trait_did: Option<ast::DefId>,
                              lhs_expr: &'a ast::Expr)
                              -> Result<Ty<'tcx>,()>
{
    debug!("lookup_op_method(expr={}, lhs_ty={}, opname={:?}, trait_did={}, lhs_expr={})",
           expr.repr(fcx.tcx()),
           lhs_ty.repr(fcx.tcx()),
           opname,
           trait_did.repr(fcx.tcx()),
           lhs_expr.repr(fcx.tcx()));

    let method = match trait_did {
        Some(trait_did) => {
            // We do eager coercions to make using operators
            // more ergonomic:
            //
            // - If the input is of type &'a T (resp. &'a mut T),
            //   then reborrow it to &'b T (resp. &'b mut T) where
            //   'b <= 'a.  This makes things like `x == y`, where
            //   `x` and `y` are both region pointers, work.  We
            //   could also solve this with variance or different
            //   traits that don't force left and right to have same
            //   type.
            let (adj_ty, adjustment) = match lhs_ty.sty {
                ty::ty_rptr(r_in, mt) => {
                    let r_adj = fcx.infcx().next_region_var(infer::Autoref(lhs_expr.span));
                    fcx.mk_subr(infer::Reborrow(lhs_expr.span), r_adj, *r_in);
                    let adjusted_ty = ty::mk_rptr(fcx.tcx(), fcx.tcx().mk_region(r_adj), mt);
                    let autoptr = ty::AutoPtr(r_adj, mt.mutbl, None);
                    let adjustment = ty::AutoDerefRef { autoderefs: 1, autoref: Some(autoptr) };
                    (adjusted_ty, adjustment)
                }
                _ => {
                    (lhs_ty, ty::AutoDerefRef { autoderefs: 0, autoref: None })
                }
            };

            debug!("adjusted_ty={} adjustment={:?}",
                   adj_ty.repr(fcx.tcx()),
                   adjustment);

            method::lookup_in_trait_adjusted(fcx, expr.span, Some(lhs_expr), opname,
                                             trait_did, adjustment, adj_ty, Some(other_tys))
        }
        None => None
    };

    match method {
        Some(method) => {
            let method_ty = method.ty;

            // HACK(eddyb) Fully qualified path to work around a resolve bug.
            let method_call = ::middle::ty::MethodCall::expr(expr.id);
            fcx.inh.method_map.borrow_mut().insert(method_call, method);

            // extract return type for method; all late bound regions
            // should have been instantiated by now
            let ret_ty = ty::ty_fn_ret(method_ty);
            Ok(ty::no_late_bound_regions(fcx.tcx(), &ret_ty).unwrap().unwrap())
        }
        None => {
            Err(())
        }
    }
}

// Binary operator categories. These categories summarize the behavior
// with respect to the builtin operationrs supported.
enum BinOpCategory {
    /// &&, || -- cannot be overridden
    Shortcircuit,

    /// <<, >> -- when shifting a single integer, rhs can be any
    /// integer type. For simd, types must match.
    Shift,

    /// +, -, etc -- takes equal types, produces same type as input,
    /// applicable to ints/floats/simd
    Math,

    /// &, |, ^ -- takes equal types, produces same type as input,
    /// applicable to ints/floats/simd/bool
    Bitwise,

    /// ==, !=, etc -- takes equal types, produces bools, except for simd,
    /// which produce the input type
    Comparison,
}

impl BinOpCategory {
    fn from(op: ast::BinOp) -> BinOpCategory {
        match op.node {
            ast::BiShl | ast::BiShr =>
                BinOpCategory::Shift,

            ast::BiAdd |
            ast::BiSub |
            ast::BiMul |
            ast::BiDiv |
            ast::BiRem =>
                BinOpCategory::Math,

            ast::BiBitXor |
            ast::BiBitAnd |
            ast::BiBitOr =>
                BinOpCategory::Bitwise,

            ast::BiEq |
            ast::BiNe |
            ast::BiLt |
            ast::BiLe |
            ast::BiGe |
            ast::BiGt =>
                BinOpCategory::Comparison,

            ast::BiAnd |
            ast::BiOr =>
                BinOpCategory::Shortcircuit,
        }
    }
}

/// Returns true if this is a built-in arithmetic operation (e.g. u32
/// + u32, i16x4 == i16x4) and false if these types would have to be
/// overloaded to be legal. There are two reasons that we distinguish
/// builtin operations from overloaded ones (vs trying to drive
/// everything uniformly through the trait system and intrinsics or
/// something like that):
///
/// 1. Builtin operations can trivially be evaluated in constants.
/// 2. For comparison operators applied to SIMD types the result is
///    not of type `bool`. For example, `i16x4==i16x4` yields a
///    type like `i16x4`. This means that the overloaded trait
///    `PartialEq` is not applicable.
///
/// Reason #2 is the killer. I tried for a while to always use
/// overloaded logic and just check the types in constants/trans after
/// the fact, and it worked fine, except for SIMD types. -nmatsakis
fn is_builtin_binop<'tcx>(cx: &ty::ctxt<'tcx>,
                          lhs: Ty<'tcx>,
                          rhs: Ty<'tcx>,
                          op: ast::BinOp)
                          -> bool
{
    match BinOpCategory::from(op) {
        BinOpCategory::Shortcircuit => {
            true
        }

        BinOpCategory::Shift => {
            ty::type_is_error(lhs) || ty::type_is_error(rhs) ||
                ty::type_is_integral(lhs) && ty::type_is_integral(rhs) ||
                ty::type_is_simd(cx, lhs) && ty::type_is_simd(cx, rhs)
        }

        BinOpCategory::Math => {
            ty::type_is_error(lhs) || ty::type_is_error(rhs) ||
                ty::type_is_integral(lhs) && ty::type_is_integral(rhs) ||
                ty::type_is_floating_point(lhs) && ty::type_is_floating_point(rhs) ||
                ty::type_is_simd(cx, lhs) && ty::type_is_simd(cx, rhs)
        }

        BinOpCategory::Bitwise => {
            ty::type_is_error(lhs) || ty::type_is_error(rhs) ||
                ty::type_is_integral(lhs) && ty::type_is_integral(rhs) ||
                ty::type_is_floating_point(lhs) && ty::type_is_floating_point(rhs) ||
                ty::type_is_simd(cx, lhs) && ty::type_is_simd(cx, rhs) ||
                ty::type_is_bool(lhs) && ty::type_is_bool(rhs)
        }

        BinOpCategory::Comparison => {
            ty::type_is_error(lhs) || ty::type_is_error(rhs) ||
                ty::type_is_scalar(lhs) && ty::type_is_scalar(rhs) ||
                ty::type_is_simd(cx, lhs) && ty::type_is_simd(cx, rhs)
        }
    }
}

