use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::paths::PathLookup;
use clippy_utils::sym;
use itertools::Itertools;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, FnDecl, Pat, PatKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::kw;
use rustc_span::{Span, Symbol};

use crate::internal_paths::{APPLICABILITY, EARLY_CONTEXT, LATE_CONTEXT, TY_CTXT};

declare_tool_lint! {
    /// ### What it does
    /// Checks if variables of some types use the usual name.
    ///
    /// ### Why is this bad?
    /// Restricting the identifiers used for common things in
    /// Clippy sources increases consistency.
    ///
    /// ### Example
    /// Check that an `rustc_errors::Applicability` variable is
    /// named either `app` or `applicability`, and not
    /// `a` or `appl`.
    pub clippy::UNUSUAL_NAMES,
    Warn,
    "commonly used concepts should use usual same variable name.",
    report_in_external_macro: true
}

declare_lint_pass!(UnusualNames => [UNUSUAL_NAMES]);

const USUAL_NAMES: [(&PathLookup, &str, &[Symbol]); 4] = [
    (
        &APPLICABILITY,
        "rustc_errors::Applicability",
        &[sym::app, sym::applicability],
    ),
    (&EARLY_CONTEXT, "rustc_lint::EarlyContext", &[sym::cx]),
    (&LATE_CONTEXT, "rustc_lint::LateContext", &[sym::cx]),
    (&TY_CTXT, "rustc_middle::ty::TyCtxt", &[sym::tcx]),
];

impl<'tcx> LateLintPass<'tcx> for UnusualNames {
    fn check_stmt(&mut self, cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'_>) {
        if let StmtKind::Let(let_stmt) = stmt.kind
            && let Some(init_expr) = let_stmt.init
        {
            check_pat_name_for_ty(cx, let_stmt.pat, cx.typeck_results().expr_ty(init_expr), "variable");
        }
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        _decl: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        _span: Span,
        def_id: LocalDefId,
    ) {
        if matches!(kind, FnKind::Closure) {
            return;
        }
        for (param, ty) in body
            .params
            .iter()
            .zip(cx.tcx.fn_sig(def_id).instantiate_identity().skip_binder().inputs())
        {
            check_pat_name_for_ty(cx, param.pat, *ty, "parameter");
        }
    }
}

fn check_pat_name_for_ty(cx: &LateContext<'_>, pat: &Pat<'_>, ty: Ty<'_>, kind: &str) {
    if let PatKind::Binding(_, _, ident, _) = pat.kind {
        let ty = ty.peel_refs();
        for (usual_ty, ty_str, usual_names) in USUAL_NAMES {
            if usual_ty.matches_ty(cx, ty)
                && !usual_names.contains(&ident.name)
                && ident.name != kw::SelfLower
                && !ident.name.as_str().starts_with('_')
            {
                let usual_names = usual_names.iter().map(|name| format!("`{name}`")).join(" or ");
                span_lint_and_help(
                    cx,
                    UNUSUAL_NAMES,
                    ident.span,
                    format!("unusual name for a {kind} of type `{ty_str}`"),
                    None,
                    format!("prefer using {usual_names}"),
                );
            }
        }
    }
}
