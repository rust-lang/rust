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
};
use middle::def_id::DefId;
use middle::ty::{Ty, HasTypeFlags, PreferMutLvalue};
use syntax::ast;
use syntax::parse::token;
use rustc_front::hir;
use rustc_front::util as hir_util;

/// Check a `a <op>= b`
pub fn check_binop_assign<'a,'tcx>(fcx: &FnCtxt<'a,'tcx>,
                                   expr: &'tcx hir::Expr,
                                   op: hir::BinOp,
                                   lhs_expr: &'tcx hir::Expr,
                                   rhs_expr: &'tcx hir::Expr)
{
    check_expr_with_lvalue_pref(fcx, lhs_expr, PreferMutLvalue);

    let lhs_ty = fcx.resolve_type_vars_if_possible(fcx.expr_ty(lhs_expr));
    let (rhs_ty, return_ty) =
        check_overloaded_binop(fcx, expr, lhs_expr, lhs_ty, rhs_expr, op, IsAssign::Yes);
    let rhs_ty = fcx.resolve_type_vars_if_possible(rhs_ty);

    if !lhs_ty.is_ty_var() && !rhs_ty.is_ty_var() && is_builtin_binop(lhs_ty, rhs_ty, op) {
        enforce_builtin_binop_types(fcx, lhs_expr, lhs_ty, rhs_expr, rhs_ty, op);
        fcx.write_nil(expr.id);
    } else {
        fcx.write_ty(expr.id, return_ty);
    }

    let tcx = fcx.tcx();
    if !tcx.expr_is_lval(lhs_expr) {
        span_err!(tcx.sess, lhs_expr.span, E0067, "invalid left-hand side expression");
    }
}

/// Check a potentially overloaded binary operator.
pub fn check_binop<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                             expr: &'tcx hir::Expr,
                             op: hir::BinOp,
                             lhs_expr: &'tcx hir::Expr,
                             rhs_expr: &'tcx hir::Expr)
{
    let tcx = fcx.ccx.tcx;

    debug!("check_binop(expr.id={}, expr={:?}, op={:?}, lhs_expr={:?}, rhs_expr={:?})",
           expr.id,
           expr,
           op,
           lhs_expr,
           rhs_expr);

    check_expr(fcx, lhs_expr);
    let lhs_ty = fcx.resolve_type_vars_if_possible(fcx.expr_ty(lhs_expr));

    match BinOpCategory::from(op) {
        BinOpCategory::Shortcircuit => {
            // && and || are a simple case.
            demand::suptype(fcx, lhs_expr.span, tcx.mk_bool(), lhs_ty);
            check_expr_coercable_to_type(fcx, rhs_expr, tcx.mk_bool());
            fcx.write_ty(expr.id, tcx.mk_bool());
        }
        _ => {
            // Otherwise, we always treat operators as if they are
            // overloaded. This is the way to be most flexible w/r/t
            // types that get inferred.
            let (rhs_ty, return_ty) =
                check_overloaded_binop(fcx, expr, lhs_expr, lhs_ty, rhs_expr, op, IsAssign::No);

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
                !lhs_ty.is_ty_var() && !rhs_ty.is_ty_var() &&
                is_builtin_binop(lhs_ty, rhs_ty, op)
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
                                         lhs_expr: &'tcx hir::Expr,
                                         lhs_ty: Ty<'tcx>,
                                         rhs_expr: &'tcx hir::Expr,
                                         rhs_ty: Ty<'tcx>,
                                         op: hir::BinOp)
                                         -> Ty<'tcx>
{
    debug_assert!(is_builtin_binop(lhs_ty, rhs_ty, op));

    let tcx = fcx.tcx();
    match BinOpCategory::from(op) {
        BinOpCategory::Shortcircuit => {
            demand::suptype(fcx, lhs_expr.span, tcx.mk_bool(), lhs_ty);
            demand::suptype(fcx, rhs_expr.span, tcx.mk_bool(), rhs_ty);
            tcx.mk_bool()
        }

        BinOpCategory::Shift => {
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
            tcx.mk_bool()
        }
    }
}

fn check_overloaded_binop<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                                    expr: &'tcx hir::Expr,
                                    lhs_expr: &'tcx hir::Expr,
                                    lhs_ty: Ty<'tcx>,
                                    rhs_expr: &'tcx hir::Expr,
                                    op: hir::BinOp,
                                    is_assign: IsAssign)
                                    -> (Ty<'tcx>, Ty<'tcx>)
{
    debug!("check_overloaded_binop(expr.id={}, lhs_ty={:?}, is_assign={:?})",
           expr.id,
           lhs_ty,
           is_assign);

    let (name, trait_def_id) = name_and_trait_def_id(fcx, op, is_assign);

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
            if !lhs_ty.references_error() {
                if let IsAssign::Yes = is_assign {
                    span_err!(fcx.tcx().sess, lhs_expr.span, E0368,
                              "binary assignment operation `{}=` cannot be applied to type `{}`",
                              hir_util::binop_to_string(op.node),
                              lhs_ty);
                } else {
                    let mut err = struct_span_err!(fcx.tcx().sess, lhs_expr.span, E0369,
                        "binary operation `{}` cannot be applied to type `{}`",
                        hir_util::binop_to_string(op.node),
                        lhs_ty);
                    let missing_trait = match op.node {
                        hir::BiAdd    => Some("std::ops::Add"),
                        hir::BiSub    => Some("std::ops::Sub"),
                        hir::BiMul    => Some("std::ops::Mul"),
                        hir::BiDiv    => Some("std::ops::Div"),
                        hir::BiRem    => Some("std::ops::Rem"),
                        hir::BiBitAnd => Some("std::ops::BitAnd"),
                        hir::BiBitOr  => Some("std::ops::BitOr"),
                        hir::BiShl    => Some("std::ops::Shl"),
                        hir::BiShr    => Some("std::ops::Shr"),
                        hir::BiEq | hir::BiNe => Some("std::cmp::PartialEq"),
                        hir::BiLt | hir::BiLe | hir::BiGt | hir::BiGe =>
                            Some("std::cmp::PartialOrd"),
                        _             => None
                    };

                    if let Some(missing_trait) = missing_trait {
                        span_note!(&mut err, lhs_expr.span,
                                   "an implementation of `{}` might be missing for `{}`",
                                    missing_trait, lhs_ty);
                    }
                    err.emit();
                }
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
                                 trait_did: Option<DefId>,
                                 ex: &'tcx hir::Expr,
                                 operand_expr: &'tcx hir::Expr,
                                 operand_ty: Ty<'tcx>,
                                 op: hir::UnOp)
                                 -> Ty<'tcx>
{
    assert!(hir_util::is_by_value_unop(op));
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

fn name_and_trait_def_id(fcx: &FnCtxt,
                         op: hir::BinOp,
                         is_assign: IsAssign)
                         -> (&'static str, Option<DefId>) {
    let lang = &fcx.tcx().lang_items;

    if let IsAssign::Yes = is_assign {
        match op.node {
            hir::BiAdd => ("add_assign", lang.add_assign_trait()),
            hir::BiSub => ("sub_assign", lang.sub_assign_trait()),
            hir::BiMul => ("mul_assign", lang.mul_assign_trait()),
            hir::BiDiv => ("div_assign", lang.div_assign_trait()),
            hir::BiRem => ("rem_assign", lang.rem_assign_trait()),
            hir::BiBitXor => ("bitxor_assign", lang.bitxor_assign_trait()),
            hir::BiBitAnd => ("bitand_assign", lang.bitand_assign_trait()),
            hir::BiBitOr => ("bitor_assign", lang.bitor_assign_trait()),
            hir::BiShl => ("shl_assign", lang.shl_assign_trait()),
            hir::BiShr => ("shr_assign", lang.shr_assign_trait()),
            hir::BiLt | hir::BiLe | hir::BiGe | hir::BiGt | hir::BiEq | hir::BiNe | hir::BiAnd |
            hir::BiOr => {
                fcx.tcx().sess.span_bug(op.span, &format!("impossible assignment operation: {}=",
                                        hir_util::binop_to_string(op.node)))
            }
        }
    } else {
        match op.node {
            hir::BiAdd => ("add", lang.add_trait()),
            hir::BiSub => ("sub", lang.sub_trait()),
            hir::BiMul => ("mul", lang.mul_trait()),
            hir::BiDiv => ("div", lang.div_trait()),
            hir::BiRem => ("rem", lang.rem_trait()),
            hir::BiBitXor => ("bitxor", lang.bitxor_trait()),
            hir::BiBitAnd => ("bitand", lang.bitand_trait()),
            hir::BiBitOr => ("bitor", lang.bitor_trait()),
            hir::BiShl => ("shl", lang.shl_trait()),
            hir::BiShr => ("shr", lang.shr_trait()),
            hir::BiLt => ("lt", lang.ord_trait()),
            hir::BiLe => ("le", lang.ord_trait()),
            hir::BiGe => ("ge", lang.ord_trait()),
            hir::BiGt => ("gt", lang.ord_trait()),
            hir::BiEq => ("eq", lang.eq_trait()),
            hir::BiNe => ("ne", lang.eq_trait()),
            hir::BiAnd | hir::BiOr => {
                fcx.tcx().sess.span_bug(op.span, "&& and || are not overloadable")
            }
        }
    }
}

fn lookup_op_method<'a, 'tcx>(fcx: &'a FnCtxt<'a, 'tcx>,
                              expr: &'tcx hir::Expr,
                              lhs_ty: Ty<'tcx>,
                              other_tys: Vec<Ty<'tcx>>,
                              opname: ast::Name,
                              trait_did: Option<DefId>,
                              lhs_expr: &'a hir::Expr)
                              -> Result<Ty<'tcx>,()>
{
    debug!("lookup_op_method(expr={:?}, lhs_ty={:?}, opname={:?}, trait_did={:?}, lhs_expr={:?})",
           expr,
           lhs_ty,
           opname,
           trait_did,
           lhs_expr);

    let method = match trait_did {
        Some(trait_did) => {
            method::lookup_in_trait_adjusted(fcx,
                                             expr.span,
                                             Some(lhs_expr),
                                             opname,
                                             trait_did,
                                             0,
                                             false,
                                             lhs_ty,
                                             Some(other_tys))
        }
        None => None
    };

    match method {
        Some(method) => {
            let method_ty = method.ty;

            // HACK(eddyb) Fully qualified path to work around a resolve bug.
            let method_call = ::middle::ty::MethodCall::expr(expr.id);
            fcx.inh.tables.borrow_mut().method_map.insert(method_call, method);

            // extract return type for method; all late bound regions
            // should have been instantiated by now
            let ret_ty = method_ty.fn_ret();
            Ok(fcx.tcx().no_late_bound_regions(&ret_ty).unwrap().unwrap())
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
    fn from(op: hir::BinOp) -> BinOpCategory {
        match op.node {
            hir::BiShl | hir::BiShr =>
                BinOpCategory::Shift,

            hir::BiAdd |
            hir::BiSub |
            hir::BiMul |
            hir::BiDiv |
            hir::BiRem =>
                BinOpCategory::Math,

            hir::BiBitXor |
            hir::BiBitAnd |
            hir::BiBitOr =>
                BinOpCategory::Bitwise,

            hir::BiEq |
            hir::BiNe |
            hir::BiLt |
            hir::BiLe |
            hir::BiGe |
            hir::BiGt =>
                BinOpCategory::Comparison,

            hir::BiAnd |
            hir::BiOr =>
                BinOpCategory::Shortcircuit,
        }
    }
}

/// Whether the binary operation is an assignment (`a += b`), or not (`a + b`)
#[derive(Clone, Copy, Debug)]
enum IsAssign {
    No,
    Yes,
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
fn is_builtin_binop<'tcx>(lhs: Ty<'tcx>,
                          rhs: Ty<'tcx>,
                          op: hir::BinOp)
                          -> bool
{
    match BinOpCategory::from(op) {
        BinOpCategory::Shortcircuit => {
            true
        }

        BinOpCategory::Shift => {
            lhs.references_error() || rhs.references_error() ||
                lhs.is_integral() && rhs.is_integral()
        }

        BinOpCategory::Math => {
            lhs.references_error() || rhs.references_error() ||
                lhs.is_integral() && rhs.is_integral() ||
                lhs.is_floating_point() && rhs.is_floating_point()
        }

        BinOpCategory::Bitwise => {
            lhs.references_error() || rhs.references_error() ||
                lhs.is_integral() && rhs.is_integral() ||
                lhs.is_floating_point() && rhs.is_floating_point() ||
                lhs.is_bool() && rhs.is_bool()
        }

        BinOpCategory::Comparison => {
            lhs.references_error() || rhs.references_error() ||
                lhs.is_scalar() && rhs.is_scalar()
        }
    }
}
