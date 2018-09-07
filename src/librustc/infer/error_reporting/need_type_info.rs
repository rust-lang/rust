// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::{self, HirId};
use infer::InferCtxt;
use infer::type_variable::TypeVariableOrigin;
use traits;
use ty::{self, Ty, Infer, TyVar};
use syntax::source_map::CompilerDesugaringKind;
use errors::DiagnosticBuilder;

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    pub fn extract_type_name(&self, ty: &'a Ty<'tcx>) -> String {
        if let ty::Infer(ty::TyVar(ty_vid)) = (*ty).sty {
            let ty_vars = self.type_variables.borrow();
            if let TypeVariableOrigin::TypeParameterDefinition(_, name) =
                *ty_vars.var_origin(ty_vid) {
                name.to_string()
            } else {
                ty.to_string()
            }
        } else {
            ty.to_string()
        }
    }

    pub fn need_type_info_err(&self,
                              cause: &traits::ObligationCause<'tcx>,
                              ty: Ty<'tcx>)
                              -> DiagnosticBuilder<'gcx> {
        let ty = self.resolve_type_vars_if_possible(&ty);
        let name = self.extract_type_name(&ty);

        let mut labels = vec![(
            cause.span,
            if &name == "_" {
                "cannot infer type".to_string()
            } else {
                format!("cannot infer type for `{}`", name)
            },
        )];
        let mut span = cause.span;

        // NB. Lower values are more preferred.
        #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
        enum LocalKind {
            ClosureArg,
            Let,
        }

        let found_local = self.in_progress_tables.and_then(|tables| {
            let tables = tables.borrow();
            let local_id_root = tables.local_id_root?;
            assert!(local_id_root.is_local());

            tables.node_types().iter().filter_map(|(&local_id, &node_ty)| {
                let node_id = self.tcx.hir.hir_to_node_id(HirId {
                    owner: local_id_root.index,
                    local_id,
                });

                let (kind, pattern) = match self.tcx.hir.find(node_id) {
                    Some(hir::Node::Local(local)) => {
                        (LocalKind::Let, &*local.pat)
                    }

                    Some(hir::Node::Binding(pat)) |
                    Some(hir::Node::Pat(pat)) => {
                        let parent_id = self.tcx.hir.get_parent_node(node_id);
                        match self.tcx.hir.find(parent_id) {
                            Some(hir::Node::Expr(e)) => {
                                match e.node {
                                    hir::ExprKind::Closure(..) => {}
                                    _ => return None,
                                }
                            }
                            _ => return None,
                        }

                        (LocalKind::ClosureArg, pat)
                    }

                    _ => return None
                };

                let node_ty = self.resolve_type_vars_if_possible(&node_ty);
                let matches_type = node_ty.walk().any(|inner_ty| {
                    inner_ty == ty || match (&inner_ty.sty, &ty.sty) {
                        (&Infer(TyVar(a_vid)), &Infer(TyVar(b_vid))) => {
                            self.type_variables
                                .borrow_mut()
                                .sub_unified(a_vid, b_vid)
                        }
                        _ => false,
                    }
                });
                if !matches_type {
                    return None;
                }

                Some((kind, pattern))
            }).min_by_key(|&(kind, pattern)| (kind, pattern.hir_id.local_id))
        });

        if let Some((LocalKind::ClosureArg, pattern)) = found_local {
            span = pattern.span;
            // We don't want to show the default label for closures.
            //
            // So, before clearing, the output would look something like this:
            // ```
            // let x = |_| {  };
            //          -  ^^^^ cannot infer type for `[_; 0]`
            //          |
            //          consider giving this closure parameter a type
            // ```
            //
            // After clearing, it looks something like this:
            // ```
            // let x = |_| {  };
            //          ^ consider giving this closure parameter a type
            // ```
            labels.clear();
            labels.push(
                (pattern.span, "consider giving this closure parameter a type".to_string()));
        } else if let Some((LocalKind::Let, pattern)) = found_local {
            if let Some(simple_ident) = pattern.simple_ident() {
                match pattern.span.compiler_desugaring_kind() {
                    None => labels.push((pattern.span,
                                         format!("consider giving `{}` a type", simple_ident))),
                    Some(CompilerDesugaringKind::ForLoop) => labels.push((
                        pattern.span,
                        "the element type for this iterator is not specified".to_string(),
                    )),
                    _ => {}
                }
            } else {
                labels.push((pattern.span, "consider giving the pattern a type".to_string()));
            }
        }

        let lint = self.get_lint_from_cause_code(&cause.code);
        macro_rules! struct_span_err_or_lint {
            ($code:ident, $($message:tt)*) => {
                match lint {
                    Some((lint, id)) => {
                        let message = format!($($message)*);
                        self.tcx.struct_span_lint_node(lint, id, span, &message)
                    }
                    None => {
                        struct_span_err!(self.tcx.sess, span, $code, $($message)*)
                    }
                }
            }
        }

        let mut err = struct_span_err_or_lint!(E0282, "type annotations needed");

        for (target_span, label_message) in labels {
            err.span_label(target_span, label_message);
        }

        err
    }
}
