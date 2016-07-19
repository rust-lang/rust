// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use check::FnCtxt;
use rustc::ty::Ty;
use rustc::infer::{InferOk, TypeOrigin};

use syntax_pos::Span;
use rustc::hir;

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    // Requires that the two types unify, and prints an error message if
    // they don't.
    pub fn demand_suptype(&self, sp: Span, expected: Ty<'tcx>, actual: Ty<'tcx>) {
        let origin = TypeOrigin::Misc(sp);
        match self.sub_types(false, origin, actual, expected) {
            Ok(InferOk { obligations, .. }) => {
                // FIXME(#32730) propagate obligations
                assert!(obligations.is_empty());
            },
            Err(e) => {
                self.report_mismatched_types(origin, expected, actual, e);
            }
        }
    }

    pub fn demand_eqtype(&self, sp: Span, expected: Ty<'tcx>, actual: Ty<'tcx>) {
        let origin = TypeOrigin::Misc(sp);
        match self.eq_types(false, origin, actual, expected) {
            Ok(InferOk { obligations, .. }) => {
                // FIXME(#32730) propagate obligations
                assert!(obligations.is_empty());
            },
            Err(e) => {
                self.report_mismatched_types(origin, expected, actual, e);
            }
        }
    }

    // Checks that the type of `expr` can be coerced to `expected`.
    pub fn demand_coerce(&self, expr: &hir::Expr, expected: Ty<'tcx>) {
        let expected = self.resolve_type_vars_with_obligations(expected);
        if let Err(e) = self.try_coerce(expr, expected) {
            let origin = TypeOrigin::Misc(expr.span);
            let expr_ty = self.resolve_type_vars_with_obligations(self.expr_ty(expr));
            self.report_mismatched_types(origin, expected, expr_ty, e);
        }
    }

    pub fn require_same_types(&self, span: Span, t1: Ty<'tcx>, t2: Ty<'tcx>, msg: &str)
                              -> bool {
        if let Err(err) = self.eq_types(false, TypeOrigin::Misc(span), t1, t2) {
            let found_ty = self.resolve_type_vars_if_possible(&t1);
            let expected_ty = self.resolve_type_vars_if_possible(&t2);
            ::emit_type_err(self.tcx, span, found_ty, expected_ty, &err, msg);
            false
        } else {
            true
        }
    }
}
