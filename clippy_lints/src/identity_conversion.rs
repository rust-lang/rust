use crate::utils::{
    in_macro_or_desugar, match_def_path, match_trait_method, same_tys, snippet, snippet_with_macro_callsite,
    span_lint_and_then,
};
use crate::utils::{paths, resolve_node};
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, impl_lint_pass};
use rustc_errors::Applicability;

declare_clippy_lint! {
    /// **What it does:** Checks for always-identical `Into`/`From`/`IntoIter` conversions.
    ///
    /// **Why is this bad?** Redundant code.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// // format!() returns a `String`
    /// let s: String = format!("hello").into();
    /// ```
    pub IDENTITY_CONVERSION,
    complexity,
    "using always-identical `Into`/`From`/`IntoIter` conversions"
}

#[derive(Default)]
pub struct IdentityConversion {
    try_desugar_arm: Vec<HirId>,
}

impl_lint_pass!(IdentityConversion => [IDENTITY_CONVERSION]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for IdentityConversion {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if in_macro_or_desugar(e.span) {
            return;
        }

        if Some(&e.hir_id) == self.try_desugar_arm.last() {
            return;
        }

        match e.node {
            ExprKind::Match(_, ref arms, MatchSource::TryDesugar) => {
                let e = match arms[0].body.node {
                    ExprKind::Ret(Some(ref e)) | ExprKind::Break(_, Some(ref e)) => e,
                    _ => return,
                };
                if let ExprKind::Call(_, ref args) = e.node {
                    self.try_desugar_arm.push(args[0].hir_id);
                }
            },

            ExprKind::MethodCall(ref name, .., ref args) => {
                if match_trait_method(cx, e, &paths::INTO) && &*name.ident.as_str() == "into" {
                    let a = cx.tables.expr_ty(e);
                    let b = cx.tables.expr_ty(&args[0]);
                    if same_tys(cx, a, b) {
                        let sugg = snippet_with_macro_callsite(cx, args[0].span, "<expr>").to_string();

                        span_lint_and_then(cx, IDENTITY_CONVERSION, e.span, "identical conversion", |db| {
                            db.span_suggestion(
                                e.span,
                                "consider removing `.into()`",
                                sugg,
                                Applicability::MachineApplicable, // snippet
                            );
                        });
                    }
                }
                if match_trait_method(cx, e, &paths::INTO_ITERATOR) && &*name.ident.as_str() == "into_iter" {
                    let a = cx.tables.expr_ty(e);
                    let b = cx.tables.expr_ty(&args[0]);
                    if same_tys(cx, a, b) {
                        let sugg = snippet(cx, args[0].span, "<expr>").into_owned();
                        span_lint_and_then(cx, IDENTITY_CONVERSION, e.span, "identical conversion", |db| {
                            db.span_suggestion(
                                e.span,
                                "consider removing `.into_iter()`",
                                sugg,
                                Applicability::MachineApplicable, // snippet
                            );
                        });
                    }
                }
            },

            ExprKind::Call(ref path, ref args) => {
                if let ExprKind::Path(ref qpath) = path.node {
                    if let Some(def_id) = resolve_node(cx, qpath, path.hir_id).opt_def_id() {
                        if match_def_path(cx, def_id, &paths::FROM_FROM) {
                            let a = cx.tables.expr_ty(e);
                            let b = cx.tables.expr_ty(&args[0]);
                            if same_tys(cx, a, b) {
                                let sugg = snippet(cx, args[0].span.source_callsite(), "<expr>").into_owned();
                                let sugg_msg =
                                    format!("consider removing `{}()`", snippet(cx, path.span, "From::from"));
                                span_lint_and_then(cx, IDENTITY_CONVERSION, e.span, "identical conversion", |db| {
                                    db.span_suggestion(
                                        e.span,
                                        &sugg_msg,
                                        sugg,
                                        Applicability::MachineApplicable, // snippet
                                    );
                                });
                            }
                        }
                    }
                }
            },

            _ => {},
        }
    }

    fn check_expr_post(&mut self, _: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if Some(&e.hir_id) == self.try_desugar_arm.last() {
            self.try_desugar_arm.pop();
        }
    }
}
