use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_macro_callsite;
use clippy_utils::visitors::for_each_value_source;
use core::ops::ControlFlow;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, PatKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, Ty, TypeFoldable, TypeVisitor};

use super::LET_UNIT_VALUE;

pub(super) fn check(cx: &LateContext<'_>, stmt: &Stmt<'_>) {
    if let StmtKind::Local(local) = stmt.kind
        && let Some(init) = local.init
        && !local.pat.span.from_expansion()
        && !in_external_macro(cx.sess(), stmt.span)
        && cx.typeck_results().pat_ty(local.pat).is_unit()
    {
        let needs_inferred = for_each_value_source(init, &mut |e| if needs_inferred_result_ty(cx, e) {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }).is_continue();

        if needs_inferred {
            if !matches!(local.pat.kind, PatKind::Wild) {
                span_lint_and_then(
                    cx,
                    LET_UNIT_VALUE,
                    stmt.span,
                    "this let-binding has unit value",
                    |diag| {
                            diag.span_suggestion(
                                local.pat.span,
                                "use a wild (`_`) binding",
                                "_",
                                Applicability::MaybeIncorrect, // snippet
                            );
                    },
                );
            }
        } else {
            span_lint_and_then(
                cx,
                LET_UNIT_VALUE,
                stmt.span,
                "this let-binding has unit value",
                |diag| {
                    if let Some(expr) = &local.init {
                        let snip = snippet_with_macro_callsite(cx, expr.span, "()");
                        diag.span_suggestion(
                            stmt.span,
                            "omit the `let` binding",
                            format!("{};", snip),
                            Applicability::MachineApplicable, // snippet
                        );
                    }
                },
            );
        }
    }
}

fn needs_inferred_result_ty(cx: &LateContext<'_>, e: &Expr<'_>) -> bool {
    let id = match e.kind {
        ExprKind::Call(
            Expr {
                kind: ExprKind::Path(ref path),
                hir_id,
                ..
            },
            _,
        ) => cx.qpath_res(path, *hir_id).opt_def_id(),
        ExprKind::MethodCall(..) => cx.typeck_results().type_dependent_def_id(e.hir_id),
        _ => return false,
    };
    if let Some(id) = id
        && let sig = cx.tcx.fn_sig(id).skip_binder()
        && let ty::Param(output_ty) = *sig.output().kind()
    {
        sig.inputs().iter().all(|&ty| !ty_contains_param(ty, output_ty.index))
    } else {
        false
    }
}

fn ty_contains_param(ty: Ty<'_>, index: u32) -> bool {
    struct Visitor(u32);
    impl<'tcx> TypeVisitor<'tcx> for Visitor {
        type BreakTy = ();
        fn visit_ty(&mut self, ty: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
            if let ty::Param(ty) = *ty.kind() {
                if ty.index == self.0 {
                    ControlFlow::BREAK
                } else {
                    ControlFlow::CONTINUE
                }
            } else {
                ty.super_visit_with(self)
            }
        }
    }
    ty.visit_with(&mut Visitor(index)).is_break()
}
