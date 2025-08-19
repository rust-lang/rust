use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{SpanRangeExt, snippet_with_applicability};
use clippy_utils::sym;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for non-octal values used to set Unix file permissions.
    ///
    /// ### Why is this bad?
    /// They will be converted into octal, creating potentially
    /// unintended file permissions.
    ///
    /// ### Example
    /// ```rust,ignore
    /// use std::fs::OpenOptions;
    /// use std::os::unix::fs::OpenOptionsExt;
    ///
    /// let mut options = OpenOptions::new();
    /// options.mode(644);
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// use std::fs::OpenOptions;
    /// use std::os::unix::fs::OpenOptionsExt;
    ///
    /// let mut options = OpenOptions::new();
    /// options.mode(0o644);
    /// ```
    #[clippy::version = "1.53.0"]
    pub NON_OCTAL_UNIX_PERMISSIONS,
    correctness,
    "use of non-octal value to set unix file permissions, which will be translated into octal"
}

declare_lint_pass!(NonOctalUnixPermissions => [NON_OCTAL_UNIX_PERMISSIONS]);

impl<'tcx> LateLintPass<'tcx> for NonOctalUnixPermissions {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        match &expr.kind {
            ExprKind::MethodCall(path, func, [param], _) => {
                if let Some(adt) = cx.typeck_results().expr_ty(func).peel_refs().ty_adt_def()
                    && matches!(
                        (cx.tcx.get_diagnostic_name(adt.did()), path.ident.name),
                        (Some(sym::FsOpenOptions | sym::DirBuilder), sym::mode)
                            | (Some(sym::FsPermissions), sym::set_mode)
                    )
                    && let ExprKind::Lit(_) = param.kind
                    && param.span.eq_ctxt(expr.span)
                    && param
                        .span
                        .check_source_text(cx, |src| !matches!(src.as_bytes(), [b'0', b'o' | b'b', ..]))
                {
                    show_error(cx, param);
                }
            },
            ExprKind::Call(func, [param]) => {
                if let ExprKind::Path(ref path) = func.kind
                    && let Some(def_id) = cx.qpath_res(path, func.hir_id).opt_def_id()
                    && cx.tcx.is_diagnostic_item(sym::permissions_from_mode, def_id)
                    && let ExprKind::Lit(_) = param.kind
                    && param.span.eq_ctxt(expr.span)
                    && param
                        .span
                        .check_source_text(cx, |src| !matches!(src.as_bytes(), [b'0', b'o' | b'b', ..]))
                {
                    show_error(cx, param);
                }
            },
            _ => {},
        }
    }
}

fn show_error(cx: &LateContext<'_>, param: &Expr<'_>) {
    let mut applicability = Applicability::MachineApplicable;
    span_lint_and_sugg(
        cx,
        NON_OCTAL_UNIX_PERMISSIONS,
        param.span,
        "using a non-octal value to set unix file permissions",
        "consider using an octal literal instead",
        format!(
            "0o{}",
            snippet_with_applicability(cx, param.span, "0o..", &mut applicability,),
        ),
        applicability,
    );
}
