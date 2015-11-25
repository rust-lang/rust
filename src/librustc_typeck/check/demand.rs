// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use check::{coercion, FnCtxt};
use middle::ty::{self, Ty};
use middle::infer::{self, TypeOrigin};

use std::result::Result::{Err, Ok};
use syntax::codemap::Span;
use rustc_front::hir;

// Requires that the two types unify, and prints an error message if
// they don't.
pub fn suptype<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>, sp: Span,
                         ty_expected: Ty<'tcx>, ty_actual: Ty<'tcx>) {
    suptype_with_fn(fcx, sp, false, ty_expected, ty_actual,
        |sp, e, a, s| { fcx.report_mismatched_types(sp, e, a, s) })
}

/// As `suptype`, but call `handle_err` if unification for subtyping fails.
pub fn suptype_with_fn<'a, 'tcx, F>(fcx: &FnCtxt<'a, 'tcx>,
                                    sp: Span,
                                    b_is_expected: bool,
                                    ty_a: Ty<'tcx>,
                                    ty_b: Ty<'tcx>,
                                    handle_err: F) where
    F: FnOnce(Span, Ty<'tcx>, Ty<'tcx>, &ty::error::TypeError<'tcx>),
{
    // n.b.: order of actual, expected is reversed
    match infer::mk_subty(fcx.infcx(), b_is_expected, TypeOrigin::Misc(sp),
                          ty_b, ty_a) {
      Ok(()) => { /* ok */ }
      Err(ref err) => {
          handle_err(sp, ty_a, ty_b, err);
      }
    }
}

pub fn eqtype<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>, sp: Span,
                        expected: Ty<'tcx>, actual: Ty<'tcx>) {
    match infer::mk_eqty(fcx.infcx(), false, TypeOrigin::Misc(sp), actual, expected) {
        Ok(()) => { /* ok */ }
        Err(ref err) => { fcx.report_mismatched_types(sp, expected, actual, err); }
    }
}

// Checks that the type of `expr` can be coerced to `expected`.
pub fn coerce<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                        sp: Span,
                        expected: Ty<'tcx>,
                        expr: &hir::Expr) {
    let expr_ty = fcx.expr_ty(expr);
    debug!("demand::coerce(expected = {:?}, expr_ty = {:?})",
           expected,
           expr_ty);
    let expr_ty = fcx.resolve_type_vars_if_possible(expr_ty);
    let expected = fcx.resolve_type_vars_if_possible(expected);
    match coercion::mk_assignty(fcx, expr, expr_ty, expected) {
      Ok(()) => { /* ok */ }
      Err(ref err) => {
        fcx.report_mismatched_types(sp, expected, expr_ty, err);
      }
    }
}
