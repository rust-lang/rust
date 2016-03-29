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
use rustc::ty::Ty;
use rustc::infer::{InferOk, TypeOrigin};

use syntax::codemap::Span;
use rustc::hir;

// Requires that the two types unify, and prints an error message if
// they don't.
pub fn suptype<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>, sp: Span,
                         expected: Ty<'tcx>, actual: Ty<'tcx>) {
    let origin = TypeOrigin::Misc(sp);
    match fcx.infcx().sub_types(false, origin, actual, expected) {
        Ok(InferOk { obligations, .. }) => {
            // FIXME(#32730) propagate obligations
            assert!(obligations.is_empty());
        },
        Err(e) => {
            fcx.infcx().report_mismatched_types(origin, expected, actual, e);
        }
    }
}

pub fn eqtype<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>, sp: Span,
                        expected: Ty<'tcx>, actual: Ty<'tcx>) {
    let origin = TypeOrigin::Misc(sp);
    match fcx.infcx().eq_types(false, origin, actual, expected) {
        Ok(InferOk { obligations, .. }) => {
            // FIXME(#32730) propagate obligations
            assert!(obligations.is_empty());
        },
        Err(e) => {
            fcx.infcx().report_mismatched_types(origin, expected, actual, e);
        }
    }
}

// Checks that the type of `expr` can be coerced to `expected`.
pub fn coerce<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                        sp: Span,
                        expected: Ty<'tcx>,
                        expr: &hir::Expr) {
    let expected = fcx.resolve_type_vars_if_possible(expected);
    if let Err(e) = coercion::try(fcx, expr, expected) {
        let origin = TypeOrigin::Misc(sp);
        let expr_ty = fcx.resolve_type_vars_if_possible(fcx.expr_ty(expr));
        fcx.infcx().report_mismatched_types(origin, expected, expr_ty, e);
    }
}
