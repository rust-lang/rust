use crate::utils::{is_copy, span_lint_and_then, sugg};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use std::iter;

use super::CLONE_DOUBLE_REF;
use super::CLONE_ON_COPY;

/// Checks for the `CLONE_ON_COPY` lint.
pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, arg: &hir::Expr<'_>, arg_ty: Ty<'_>) {
    let ty = cx.typeck_results().expr_ty(expr);
    if let ty::Ref(_, inner, _) = arg_ty.kind() {
        if let ty::Ref(_, innermost, _) = inner.kind() {
            span_lint_and_then(
                cx,
                CLONE_DOUBLE_REF,
                expr.span,
                &format!(
                    "using `clone` on a double-reference; \
                    this will copy the reference of type `{}` instead of cloning the inner type",
                    ty
                ),
                |diag| {
                    if let Some(snip) = sugg::Sugg::hir_opt(cx, arg) {
                        let mut ty = innermost;
                        let mut n = 0;
                        while let ty::Ref(_, inner, _) = ty.kind() {
                            ty = inner;
                            n += 1;
                        }
                        let refs: String = iter::repeat('&').take(n + 1).collect();
                        let derefs: String = iter::repeat('*').take(n).collect();
                        let explicit = format!("<{}{}>::clone({})", refs, ty, snip);
                        diag.span_suggestion(
                            expr.span,
                            "try dereferencing it",
                            format!("{}({}{}).clone()", refs, derefs, snip.deref()),
                            Applicability::MaybeIncorrect,
                        );
                        diag.span_suggestion(
                            expr.span,
                            "or try being explicit if you are sure, that you want to clone a reference",
                            explicit,
                            Applicability::MaybeIncorrect,
                        );
                    }
                },
            );
            return; // don't report clone_on_copy
        }
    }

    if is_copy(cx, ty) {
        let snip;
        if let Some(snippet) = sugg::Sugg::hir_opt(cx, arg) {
            let parent = cx.tcx.hir().get_parent_node(expr.hir_id);
            match &cx.tcx.hir().get(parent) {
                hir::Node::Expr(parent) => match parent.kind {
                    // &*x is a nop, &x.clone() is not
                    hir::ExprKind::AddrOf(..) => return,
                    // (*x).func() is useless, x.clone().func() can work in case func borrows mutably
                    hir::ExprKind::MethodCall(_, _, parent_args, _) if expr.hir_id == parent_args[0].hir_id => {
                        return;
                    },

                    _ => {},
                },
                hir::Node::Stmt(stmt) => {
                    if let hir::StmtKind::Local(ref loc) = stmt.kind {
                        if let hir::PatKind::Ref(..) = loc.pat.kind {
                            // let ref y = *x borrows x, let ref y = x.clone() does not
                            return;
                        }
                    }
                },
                _ => {},
            }

            // x.clone() might have dereferenced x, possibly through Deref impls
            if cx.typeck_results().expr_ty(arg) == ty {
                snip = Some(("try removing the `clone` call", format!("{}", snippet)));
            } else {
                let deref_count = cx
                    .typeck_results()
                    .expr_adjustments(arg)
                    .iter()
                    .filter(|adj| matches!(adj.kind, ty::adjustment::Adjust::Deref(_)))
                    .count();
                let derefs: String = iter::repeat('*').take(deref_count).collect();
                snip = Some(("try dereferencing it", format!("{}{}", derefs, snippet)));
            }
        } else {
            snip = None;
        }
        span_lint_and_then(
            cx,
            CLONE_ON_COPY,
            expr.span,
            &format!("using `clone` on type `{}` which implements the `Copy` trait", ty),
            |diag| {
                if let Some((text, snip)) = snip {
                    diag.span_suggestion(expr.span, text, snip, Applicability::MachineApplicable);
                }
            },
        );
    }
}
