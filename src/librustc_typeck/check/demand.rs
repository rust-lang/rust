// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::iter;

use check::FnCtxt;
use rustc::infer::InferOk;
use rustc::traits::ObligationCause;

use syntax::ast;
use syntax_pos::{self, Span};
use rustc::hir;
use rustc::hir::print;
use rustc::hir::def::Def;
use rustc::ty::{self, Ty, AssociatedItem};
use errors::{DiagnosticBuilder, CodeMapper};

use super::method::probe;

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    // Requires that the two types unify, and prints an error message if
    // they don't.
    pub fn demand_suptype(&self, sp: Span, expected: Ty<'tcx>, actual: Ty<'tcx>) {
        self.demand_suptype_diag(sp, expected, actual).map(|mut e| e.emit());
    }

    pub fn demand_suptype_diag(&self,
                               sp: Span,
                               expected: Ty<'tcx>,
                               actual: Ty<'tcx>) -> Option<DiagnosticBuilder<'tcx>> {
        let cause = &self.misc(sp);
        match self.at(cause, self.param_env).sup(expected, actual) {
            Ok(InferOk { obligations, value: () }) => {
                self.register_predicates(obligations);
                None
            },
            Err(e) => {
                Some(self.report_mismatched_types(&cause, expected, actual, e))
            }
        }
    }

    pub fn demand_eqtype(&self, sp: Span, expected: Ty<'tcx>, actual: Ty<'tcx>) {
        if let Some(mut err) = self.demand_eqtype_diag(sp, expected, actual) {
            err.emit();
        }
    }

    pub fn demand_eqtype_diag(&self,
                             sp: Span,
                             expected: Ty<'tcx>,
                             actual: Ty<'tcx>) -> Option<DiagnosticBuilder<'tcx>> {
        self.demand_eqtype_with_origin(&self.misc(sp), expected, actual)
    }

    pub fn demand_eqtype_with_origin(&self,
                                     cause: &ObligationCause<'tcx>,
                                     expected: Ty<'tcx>,
                                     actual: Ty<'tcx>) -> Option<DiagnosticBuilder<'tcx>> {
        match self.at(cause, self.param_env).eq(expected, actual) {
            Ok(InferOk { obligations, value: () }) => {
                self.register_predicates(obligations);
                None
            }
            Err(e) => {
                Some(self.report_mismatched_types(cause, expected, actual, e))
            }
        }
    }

    pub fn demand_coerce(&self,
                         expr: &hir::Expr,
                         checked_ty: Ty<'tcx>,
                         expected: Ty<'tcx>)
                         -> Ty<'tcx> {
        let (ty, err) = self.demand_coerce_diag(expr, checked_ty, expected);
        if let Some(mut err) = err {
            err.emit();
        }
        ty
    }

    // Checks that the type of `expr` can be coerced to `expected`.
    //
    // NB: This code relies on `self.diverges` to be accurate. In
    // particular, assignments to `!` will be permitted if the
    // diverges flag is currently "always".
    pub fn demand_coerce_diag(&self,
                              expr: &hir::Expr,
                              checked_ty: Ty<'tcx>,
                              expected: Ty<'tcx>)
                              -> (Ty<'tcx>, Option<DiagnosticBuilder<'tcx>>) {
        let expected = self.resolve_type_vars_with_obligations(expected);

        let e = match self.try_coerce(expr, checked_ty, self.diverges.get(), expected) {
            Ok(ty) => return (ty, None),
            Err(e) => e
        };

        let cause = self.misc(expr.span);
        let expr_ty = self.resolve_type_vars_with_obligations(checked_ty);
        let mut err = self.report_mismatched_types(&cause, expected, expr_ty, e);

        // If the expected type is an enum with any variants whose sole
        // field is of the found type, suggest such variants. See Issue
        // #42764.
        if let ty::TyAdt(expected_adt, substs) = expected.sty {
            let mut compatible_variants = vec![];
            for variant in &expected_adt.variants {
                if variant.fields.len() == 1 {
                    let sole_field = &variant.fields[0];
                    let sole_field_ty = sole_field.ty(self.tcx, substs);
                    if self.can_coerce(expr_ty, sole_field_ty) {
                        let mut variant_path = self.tcx.item_path_str(variant.did);
                        variant_path = variant_path.trim_left_matches("std::prelude::v1::")
                            .to_string();
                        compatible_variants.push(variant_path);
                    }
                }
            }
            if !compatible_variants.is_empty() {
                let expr_text = print::to_string(print::NO_ANN, |s| s.print_expr(expr));
                let suggestions = compatible_variants.iter()
                    .map(|v| format!("{}({})", v, expr_text)).collect::<Vec<_>>();
                err.span_suggestions(expr.span,
                                     "try using a variant of the expected type",
                                     suggestions);
            }
        }

        if let Some((msg, suggestion)) = self.check_ref(expr, checked_ty, expected) {
            err.span_suggestion(expr.span, msg, suggestion);
        } else {
            let methods = self.get_conversion_methods(expected, checked_ty);
            if let Ok(expr_text) = self.tcx.sess.codemap().span_to_snippet(expr.span) {
                let suggestions = iter::repeat(expr_text).zip(methods.iter())
                    .map(|(receiver, method)| format!("{}.{}()", receiver, method.name))
                    .collect::<Vec<_>>();
                if !suggestions.is_empty() {
                    err.span_suggestions(expr.span,
                                         "try using a conversion method",
                                         suggestions);
                }
            }
        }
        (expected, Some(err))
    }

    fn get_conversion_methods(&self, expected: Ty<'tcx>, checked_ty: Ty<'tcx>)
                              -> Vec<AssociatedItem> {
        let mut methods = self.probe_for_return_type(syntax_pos::DUMMY_SP,
                                                     probe::Mode::MethodCall,
                                                     expected,
                                                     checked_ty,
                                                     ast::DUMMY_NODE_ID);
        methods.retain(|m| {
            self.has_no_input_arg(m) &&
                self.tcx.get_attrs(m.def_id).iter()
                // This special internal attribute is used to whitelist
                // "identity-like" conversion methods to be suggested here.
                //
                // FIXME (#46459 and #46460): ideally
                // `std::convert::Into::into` and `std::borrow:ToOwned` would
                // also be `#[rustc_conversion_suggestion]`, if not for
                // method-probing false-positives and -negatives (respectively).
                //
                // FIXME? Other potential candidate methods: `as_ref` and
                // `as_mut`?
                .find(|a| a.check_name("rustc_conversion_suggestion")).is_some()
        });

        methods
    }

    // This function checks if the method isn't static and takes other arguments than `self`.
    fn has_no_input_arg(&self, method: &AssociatedItem) -> bool {
        match method.def() {
            Def::Method(def_id) => {
                self.tcx.fn_sig(def_id).inputs().skip_binder().len() == 1
            }
            _ => false,
        }
    }

    /// This function is used to determine potential "simple" improvements or users' errors and
    /// provide them useful help. For example:
    ///
    /// ```
    /// fn some_fn(s: &str) {}
    ///
    /// let x = "hey!".to_owned();
    /// some_fn(x); // error
    /// ```
    ///
    /// No need to find every potential function which could make a coercion to transform a
    /// `String` into a `&str` since a `&` would do the trick!
    ///
    /// In addition of this check, it also checks between references mutability state. If the
    /// expected is mutable but the provided isn't, maybe we could just say "Hey, try with
    /// `&mut`!".
    fn check_ref(&self,
                 expr: &hir::Expr,
                 checked_ty: Ty<'tcx>,
                 expected: Ty<'tcx>)
                 -> Option<(&'static str, String)> {
        match (&expected.sty, &checked_ty.sty) {
            (&ty::TyRef(_, exp), &ty::TyRef(_, check)) => match (&exp.ty.sty, &check.ty.sty) {
                (&ty::TyStr, &ty::TyArray(arr, _)) |
                (&ty::TyStr, &ty::TySlice(arr)) if arr == self.tcx.types.u8 => {
                    if let hir::ExprLit(_) = expr.node {
                        let sp = self.sess().codemap().call_span_if_macro(expr.span);
                        if let Ok(src) = self.tcx.sess.codemap().span_to_snippet(sp) {
                            return Some(("consider removing the leading `b`",
                                         src[1..].to_string()));
                        }
                    }
                    None
                },
                (&ty::TyArray(arr, _), &ty::TyStr) |
                (&ty::TySlice(arr), &ty::TyStr) if arr == self.tcx.types.u8 => {
                    if let hir::ExprLit(_) = expr.node {
                        let sp = self.sess().codemap().call_span_if_macro(expr.span);
                        if let Ok(src) = self.tcx.sess.codemap().span_to_snippet(sp) {
                            return Some(("consider adding a leading `b`",
                                         format!("b{}", src)));
                        }
                    }
                    None
                }
                _ => None,
            },
            (&ty::TyRef(_, mutability), _) => {
                // Check if it can work when put into a ref. For example:
                //
                // ```
                // fn bar(x: &mut i32) {}
                //
                // let x = 0u32;
                // bar(&x); // error, expected &mut
                // ```
                let ref_ty = match mutability.mutbl {
                    hir::Mutability::MutMutable => self.tcx.mk_mut_ref(
                                                       self.tcx.mk_region(ty::ReStatic),
                                                       checked_ty),
                    hir::Mutability::MutImmutable => self.tcx.mk_imm_ref(
                                                       self.tcx.mk_region(ty::ReStatic),
                                                       checked_ty),
                };
                if self.can_coerce(ref_ty, expected) {
                    // Use the callsite's span if this is a macro call. #41858
                    let sp = self.sess().codemap().call_span_if_macro(expr.span);
                    if let Ok(src) = self.tcx.sess.codemap().span_to_snippet(sp) {
                        let sugg_expr = match expr.node { // parenthesize if needed (Issue #46756)
                            hir::ExprCast(_, _) | hir::ExprBinary(_, _, _) => format!("({})", src),
                            _ => src,
                        };
                        return Some(match mutability.mutbl {
                            hir::Mutability::MutMutable => {
                                ("consider mutably borrowing here", format!("&mut {}", sugg_expr))
                            }
                            hir::Mutability::MutImmutable => {
                                ("consider borrowing here", format!("&{}", sugg_expr))
                            }
                        });
                    }
                }
                None
            }
            (_, &ty::TyRef(_, checked)) => {
                // We have `&T`, check if what was expected was `T`. If so,
                // we may want to suggest adding a `*`, or removing
                // a `&`.
                //
                // (But, also check check the `expn_info()` to see if this is
                // a macro; if so, it's hard to extract the text and make a good
                // suggestion, so don't bother.)
                if self.infcx.can_sub(self.param_env, checked.ty, &expected).is_ok() &&
                   expr.span.ctxt().outer().expn_info().is_none() {
                    match expr.node {
                        // Maybe remove `&`?
                        hir::ExprAddrOf(_, ref expr) => {
                            if let Ok(code) = self.tcx.sess.codemap().span_to_snippet(expr.span) {
                                return Some(("consider removing the borrow",
                                             code));
                            }
                        }

                        // Maybe add `*`? Only if `T: Copy`.
                        _ => {
                            if !self.infcx.type_moves_by_default(self.param_env,
                                                                checked.ty,
                                                                expr.span) {
                                let sp = self.sess().codemap().call_span_if_macro(expr.span);
                                if let Ok(code) = self.tcx.sess.codemap().span_to_snippet(sp) {
                                    return Some(("consider dereferencing the borrow",
                                                 format!("*{}", code)));
                                }
                            }
                        },
                    }
                }
                None
            }
            _ => None,
        }
    }
}
