use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use clippy_utils::{last_path_segment, std_or_core};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, GenericArg, QPath, TyKind, def};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::{SyntaxContext, sym};

declare_clippy_lint! {
    /// ### What it does
    /// It checks for `std::iter::Empty::default()` and suggests replacing it with
    /// `std::iter::empty()`.
    /// ### Why is this bad?
    /// `std::iter::empty()` is the more idiomatic way.
    /// ### Example
    /// ```no_run
    /// let _ = std::iter::Empty::<usize>::default();
    /// let iter: std::iter::Empty<usize> = std::iter::Empty::default();
    /// ```
    /// Use instead:
    /// ```no_run
    /// let _ = std::iter::empty::<usize>();
    /// let iter: std::iter::Empty<usize> = std::iter::empty();
    /// ```
    #[clippy::version = "1.64.0"]
    pub DEFAULT_INSTEAD_OF_ITER_EMPTY,
    style,
    "check `std::iter::Empty::default()` and replace with `std::iter::empty()`"
}
declare_lint_pass!(DefaultIterEmpty => [DEFAULT_INSTEAD_OF_ITER_EMPTY]);

impl<'tcx> LateLintPass<'tcx> for DefaultIterEmpty {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Call(iter_expr, []) = &expr.kind
            && let ExprKind::Path(QPath::TypeRelative(ty, _)) = &iter_expr.kind
            && let TyKind::Path(ty_path) = &ty.kind
            && let QPath::Resolved(None, path) = ty_path
            && let def::Res::Def(_, def_id) = &path.res
            && cx.tcx.is_diagnostic_item(sym::IterEmpty, *def_id)
            && let ctxt = expr.span.ctxt()
            && ty.span.ctxt() == ctxt
        {
            let mut applicability = Applicability::MachineApplicable;
            let Some(path) = std_or_core(cx) else { return };
            let path = format!("{path}::iter::empty");
            let sugg = make_sugg(cx, ty_path, ctxt, &mut applicability, &path);
            span_lint_and_sugg(
                cx,
                DEFAULT_INSTEAD_OF_ITER_EMPTY,
                expr.span,
                format!("`{path}()` is the more idiomatic way"),
                "try",
                sugg,
                applicability,
            );
        }
    }
}

fn make_sugg(
    cx: &LateContext<'_>,
    ty_path: &QPath<'_>,
    ctxt: SyntaxContext,
    applicability: &mut Applicability,
    path: &str,
) -> String {
    if let Some(last) = last_path_segment(ty_path).args
        && let Some(iter_ty) = last.args.iter().find_map(|arg| match arg {
            GenericArg::Type(ty) => Some(ty),
            _ => None,
        })
    {
        format!(
            "{path}::<{}>()",
            snippet_with_context(cx, iter_ty.span, ctxt, "..", applicability).0
        )
    } else {
        format!("{path}()")
    }
}
