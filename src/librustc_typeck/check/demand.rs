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

use syntax::ast;
use syntax_pos::{self, Span};
use rustc::hir;
use rustc::ty::{self, ImplOrTraitItem};

use hir::def_id::DefId;

use std::rc::Rc;

use super::method::probe;

struct MethodInfo<'tcx> {
    ast: Option<ast::Attribute>,
    id: DefId,
    item: Rc<ImplOrTraitItem<'tcx>>,
}

impl<'tcx> MethodInfo<'tcx> {
    fn new(ast: Option<ast::Attribute>, id: DefId, item: Rc<ImplOrTraitItem<'tcx>>) -> MethodInfo {
        MethodInfo {
            ast: ast,
            id: id,
            item: item,
        }
    }
}

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
            let mode = probe::Mode::MethodCall;
            if let Ok(methods) = self.probe_return(syntax_pos::DUMMY_SP, mode, expected,
                                                   checked_ty, ast::DUMMY_NODE_ID) {
                let suggestions: Vec<_> =
                    methods.iter()
                           .filter_map(|ref x| {
                            if let Some(id) = self.get_impl_id(&x.item) {
                                Some(MethodInfo::new(None, id, Rc::new(x.item.clone())))
                            } else {
                                None
                            }})
                           .collect();
                let safe_suggestions: Vec<_> =
                    suggestions.iter()
                               .map(|ref x| MethodInfo::new(
                                                self.find_attr(x.id, "safe_suggestion"),
                                                               x.id,
                                                               x.item.clone()))
                               .filter(|ref x| x.ast.is_some())
                               .collect();
                if safe_suggestions.len() > 0 {
                    self.get_best_match(&safe_suggestions);
                } else {
                    self.get_best_match(&suggestions);
                }
            }
            self.report_mismatched_types(&cause, expected, expr_ty, e);
        }
    }

    fn get_best_match(&self, methods: &[MethodInfo<'tcx>]) -> String {
        if methods.len() == 1 {
            println!("unique match ==> {:?}", methods[0].item.name());
            return String::new();
        }
        let no_argument_methods: Vec<&MethodInfo> =
            methods.iter()
                   .filter(|ref x| self.has_not_input_arg(&*x.item))
                   .collect();
        if no_argument_methods.len() > 0 {
            for ref method in no_argument_methods {
                println!("best match ==> {:?}", method.item.name());
            }
        } else {
            for ref method in methods.iter() {
                println!("not best ==> {:?}", method.item.name());
            }
        }
        String::new()
    }

    fn get_impl_id(&self, impl_: &ImplOrTraitItem<'tcx>) -> Option<DefId> {
        match *impl_ {
            ty::ImplOrTraitItem::MethodTraitItem(ref m) => Some((*m).def_id),
            _ => None,
        }
    }

    fn has_not_input_arg(&self, method: &ImplOrTraitItem<'tcx>) -> bool {
        match *method {
            ImplOrTraitItem::MethodTraitItem(ref x) => {
                x.fty.sig.skip_binder().inputs.len() == 1
            }
            _ => false,
        }
    }
}
