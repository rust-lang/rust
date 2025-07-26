use clippy_utils::diagnostics::span_lint_and_then;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks usage of `std::fs::create_dir` and suggest using `std::fs::create_dir_all` instead.
    ///
    /// ### Why restrict this?
    /// Sometimes `std::fs::create_dir` is mistakenly chosen over `std::fs::create_dir_all`,
    /// resulting in failure when more than one directory needs to be created or when the directory already exists.
    /// Crates which never need to specifically create a single directory may wish to prevent this mistake.
    ///
    /// ### Example
    /// ```rust,ignore
    /// std::fs::create_dir("foo");
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// std::fs::create_dir_all("foo");
    /// ```
    #[clippy::version = "1.48.0"]
    pub CREATE_DIR,
    restriction,
    "calling `std::fs::create_dir` instead of `std::fs::create_dir_all`"
}

declare_lint_pass!(CreateDir => [CREATE_DIR]);

impl LateLintPass<'_> for CreateDir {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::Call(func, [_]) = expr.kind
            && let ExprKind::Path(ref path) = func.kind
            && let Some(def_id) = cx.qpath_res(path, func.hir_id).opt_def_id()
            && cx.tcx.is_diagnostic_item(sym::fs_create_dir, def_id)
            && let QPath::Resolved(_, path) = path
            && let Some(last) = path.segments.last()
        {
            span_lint_and_then(
                cx,
                CREATE_DIR,
                expr.span,
                "calling `std::fs::create_dir` where there may be a better way",
                |diag| {
                    let mut suggestions = vec![(last.ident.span.shrink_to_hi(), "_all".to_owned())];
                    if path.segments.len() == 1 {
                        suggestions.push((path.span.shrink_to_lo(), "std::fs::".to_owned()));
                    }

                    diag.multipart_suggestion_verbose(
                        "consider calling `std::fs::create_dir_all` instead",
                        suggestions,
                        Applicability::MaybeIncorrect,
                    );
                },
            );
        }
    }
}
