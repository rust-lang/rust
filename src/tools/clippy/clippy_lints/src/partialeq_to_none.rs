use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_res_lang_ctor, path_res, peel_hir_expr_refs, peel_ref_operators, sugg};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, LangItem};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Checks for binary comparisons to a literal `Option::None`.
    ///
    /// ### Why is this bad?
    ///
    /// A programmer checking if some `foo` is `None` via a comparison `foo == None`
    /// is usually inspired from other programming languages (e.g. `foo is None`
    /// in Python).
    /// Checking if a value of type `Option<T>` is (not) equal to `None` in that
    /// way relies on `T: PartialEq` to do the comparison, which is unneeded.
    ///
    /// ### Example
    /// ```rust
    /// fn foo(f: Option<u32>) -> &'static str {
    ///     if f != None { "yay" } else { "nay" }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn foo(f: Option<u32>) -> &'static str {
    ///     if f.is_some() { "yay" } else { "nay" }
    /// }
    /// ```
    #[clippy::version = "1.65.0"]
    pub PARTIALEQ_TO_NONE,
    style,
    "Binary comparison to `Option<T>::None` relies on `T: PartialEq`, which is unneeded"
}
declare_lint_pass!(PartialeqToNone => [PARTIALEQ_TO_NONE]);

impl<'tcx> LateLintPass<'tcx> for PartialeqToNone {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        // Skip expanded code, as we have no control over it anyway...
        if e.span.from_expansion() {
            return;
        }

        // If the expression is of type `Option`
        let is_ty_option =
            |expr: &Expr<'_>| is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(expr).peel_refs(), sym::Option);

        // If the expression is a literal `Option::None`
        let is_none_ctor = |expr: &Expr<'_>| {
            !expr.span.from_expansion()
                && is_res_lang_ctor(cx, path_res(cx, peel_hir_expr_refs(expr).0), LangItem::OptionNone)
        };

        let mut applicability = Applicability::MachineApplicable;

        if let ExprKind::Binary(op, left_side, right_side) = e.kind {
            // All other comparisons (e.g. `>= None`) have special meaning wrt T
            let is_eq = match op.node {
                BinOpKind::Eq => true,
                BinOpKind::Ne => false,
                _ => return,
            };

            // We are only interested in comparisons between `Option` and a literal `Option::None`
            let scrutinee = match (
                is_none_ctor(left_side) && is_ty_option(right_side),
                is_none_ctor(right_side) && is_ty_option(left_side),
            ) {
                (true, false) => right_side,
                (false, true) => left_side,
                _ => return,
            };

            // Peel away refs/derefs (as long as we don't cross manual deref impls), as
            // autoref/autoderef will take care of those
            let sugg = format!(
                "{}.{}",
                sugg::Sugg::hir_with_applicability(cx, peel_ref_operators(cx, scrutinee), "..", &mut applicability)
                    .maybe_par(),
                if is_eq { "is_none()" } else { "is_some()" }
            );

            span_lint_and_sugg(
                cx,
                PARTIALEQ_TO_NONE,
                e.span,
                "binary comparison to literal `Option::None`",
                if is_eq {
                    "use `Option::is_none()` instead"
                } else {
                    "use `Option::is_some()` instead"
                },
                sugg,
                applicability,
            );
        }
    }
}
