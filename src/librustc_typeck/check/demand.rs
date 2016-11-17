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
use rustc::infer::{InferOk};
use rustc::traits::ObligationCause;

use syntax_pos::Span;
use rustc::hir;

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    // Requires that the two types unify, and prints an error message if
    // they don't.
    pub fn demand_suptype(&self, sp: Span, expected: Ty<'tcx>, actual: Ty<'tcx>) {
        let cause = self.misc(sp);
        match self.sub_types(false, &cause, actual, expected) {
            Ok(InferOk { obligations, value: () }) => {
                self.register_predicates(obligations);
            },
            Err(e) => {
                self.report_mismatched_types(&cause, expected, actual, e);
            }
        }
    }

    pub fn demand_eqtype(&self, sp: Span, expected: Ty<'tcx>, actual: Ty<'tcx>) {
        self.demand_eqtype_with_origin(&self.misc(sp), expected, actual);
    }

    pub fn demand_eqtype_with_origin(&self,
                                     cause: &ObligationCause<'tcx>,
                                     expected: Ty<'tcx>,
                                     actual: Ty<'tcx>)
    {
        match self.eq_types(false, cause, actual, expected) {
            Ok(InferOk { obligations, value: () }) => {
                self.register_predicates(obligations);
            },
            Err(e) => {
                self.report_mismatched_types(cause, expected, actual, e);
            }
        }
    }

    // Checks that the type of `expr` can be coerced to `expected`.
    pub fn demand_coerce(&self, expr: &hir::Expr, checked_ty: Ty<'tcx>, expected: Ty<'tcx>) {
        let expected = self.resolve_type_vars_with_obligations(expected);
        if let Err(e) = self.try_coerce(expr, checked_ty, expected) {
            let cause = self.misc(expr.span);
            let expr_ty = self.resolve_type_vars_with_obligations(checked_ty);
            self.report_mismatched_types(&cause, expected, expr_ty, e);
        }
    }
}
