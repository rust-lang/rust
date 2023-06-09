use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::{is_trait_method, match_def_path, paths, peel_hir_expr_refs};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Expr, ExprKind, Mutability, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for references on `std::path::MAIN_SEPARATOR.to_string()` used
    /// to build a `&str`.
    ///
    /// ### Why is this bad?
    /// There exists a `std::path::MAIN_SEPARATOR_STR` which does not require
    /// an extra memory allocation.
    ///
    /// ### Example
    /// ```rust
    /// let s: &str = &std::path::MAIN_SEPARATOR.to_string();
    /// ```
    /// Use instead:
    /// ```rust
    /// let s: &str = std::path::MAIN_SEPARATOR_STR;
    /// ```
    #[clippy::version = "1.70.0"]
    pub MANUAL_MAIN_SEPARATOR_STR,
    complexity,
    "`&std::path::MAIN_SEPARATOR.to_string()` can be replaced by `std::path::MAIN_SEPARATOR_STR`"
}

pub struct ManualMainSeparatorStr {
    msrv: Msrv,
}

impl ManualMainSeparatorStr {
    #[must_use]
    pub fn new(msrv: Msrv) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(ManualMainSeparatorStr => [MANUAL_MAIN_SEPARATOR_STR]);

impl LateLintPass<'_> for ManualMainSeparatorStr {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if self.msrv.meets(msrvs::PATH_MAIN_SEPARATOR_STR) &&
            let (target, _) = peel_hir_expr_refs(expr) &&
            is_trait_method(cx, target, sym::ToString) &&
            let ExprKind::MethodCall(path, receiver, &[], _) = target.kind &&
            path.ident.name == sym::to_string &&
            let ExprKind::Path(QPath::Resolved(None, path)) = receiver.kind &&
            let Res::Def(DefKind::Const, receiver_def_id) = path.res &&
            match_def_path(cx, receiver_def_id, &paths::PATH_MAIN_SEPARATOR) &&
            let ty::Ref(_, ty, Mutability::Not) = cx.typeck_results().expr_ty_adjusted(expr).kind() &&
            ty.is_str()
            {
                span_lint_and_sugg(
                    cx,
                    MANUAL_MAIN_SEPARATOR_STR,
                    expr.span,
                    "taking a reference on `std::path::MAIN_SEPARATOR` conversion to `String`",
                    "replace with",
                    "std::path::MAIN_SEPARATOR_STR".to_owned(),
                    Applicability::MachineApplicable,
                );
            }
    }

    extract_msrv_attr!(LateContext);
}
