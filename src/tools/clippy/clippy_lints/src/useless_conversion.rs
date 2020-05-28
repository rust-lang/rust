use crate::utils::{
    match_def_path, match_trait_method, paths, same_tys, snippet, snippet_with_macro_callsite, span_lint_and_sugg,
};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, HirId, MatchSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};

declare_clippy_lint! {
    /// **What it does:** Checks for `Into`/`From`/`IntoIter` calls that useless converts
    /// to the same type as caller.
    ///
    /// **Why is this bad?** Redundant code.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// // Bad
    /// // format!() returns a `String`
    /// let s: String = format!("hello").into();
    ///
    /// // Good
    /// let s: String = format!("hello");
    /// ```
    pub USELESS_CONVERSION,
    complexity,
    "calls to `Into`/`From`/`IntoIter` that performs useless conversions to the same type"
}

#[derive(Default)]
pub struct UselessConversion {
    try_desugar_arm: Vec<HirId>,
}

impl_lint_pass!(UselessConversion => [USELESS_CONVERSION]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UselessConversion {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr<'_>) {
        if e.span.from_expansion() {
            return;
        }

        if Some(&e.hir_id) == self.try_desugar_arm.last() {
            return;
        }

        match e.kind {
            ExprKind::Match(_, ref arms, MatchSource::TryDesugar) => {
                let e = match arms[0].body.kind {
                    ExprKind::Ret(Some(ref e)) | ExprKind::Break(_, Some(ref e)) => e,
                    _ => return,
                };
                if let ExprKind::Call(_, ref args) = e.kind {
                    self.try_desugar_arm.push(args[0].hir_id);
                }
            },

            ExprKind::MethodCall(ref name, .., ref args) => {
                if match_trait_method(cx, e, &paths::INTO) && &*name.ident.as_str() == "into" {
                    let a = cx.tables.expr_ty(e);
                    let b = cx.tables.expr_ty(&args[0]);
                    if same_tys(cx, a, b) {
                        let sugg = snippet_with_macro_callsite(cx, args[0].span, "<expr>").to_string();

                        span_lint_and_sugg(
                            cx,
                            USELESS_CONVERSION,
                            e.span,
                            "useless conversion",
                            "consider removing `.into()`",
                            sugg,
                            Applicability::MachineApplicable, // snippet
                        );
                    }
                }
                if match_trait_method(cx, e, &paths::INTO_ITERATOR) && &*name.ident.as_str() == "into_iter" {
                    let a = cx.tables.expr_ty(e);
                    let b = cx.tables.expr_ty(&args[0]);
                    if same_tys(cx, a, b) {
                        let sugg = snippet(cx, args[0].span, "<expr>").into_owned();
                        span_lint_and_sugg(
                            cx,
                            USELESS_CONVERSION,
                            e.span,
                            "useless conversion",
                            "consider removing `.into_iter()`",
                            sugg,
                            Applicability::MachineApplicable, // snippet
                        );
                    }
                }
            },

            ExprKind::Call(ref path, ref args) => {
                if let ExprKind::Path(ref qpath) = path.kind {
                    if let Some(def_id) = cx.tables.qpath_res(qpath, path.hir_id).opt_def_id() {
                        if match_def_path(cx, def_id, &paths::FROM_FROM) {
                            let a = cx.tables.expr_ty(e);
                            let b = cx.tables.expr_ty(&args[0]);
                            if same_tys(cx, a, b) {
                                let sugg = snippet(cx, args[0].span.source_callsite(), "<expr>").into_owned();
                                let sugg_msg =
                                    format!("consider removing `{}()`", snippet(cx, path.span, "From::from"));
                                span_lint_and_sugg(
                                    cx,
                                    USELESS_CONVERSION,
                                    e.span,
                                    "useless conversion",
                                    &sugg_msg,
                                    sugg,
                                    Applicability::MachineApplicable, // snippet
                                );
                            }
                        }
                    }
                }
            },

            _ => {},
        }
    }

    fn check_expr_post(&mut self, _: &LateContext<'a, 'tcx>, e: &'tcx Expr<'_>) {
        if Some(&e.hir_id) == self.try_desugar_arm.last() {
            self.try_desugar_arm.pop();
        }
    }
}
