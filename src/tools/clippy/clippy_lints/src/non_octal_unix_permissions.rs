use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{snippet_opt, snippet_with_applicability};
use clippy_utils::ty::match_type;
use clippy_utils::{match_def_path, paths};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

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
                let obj_ty = cx.typeck_results().expr_ty(func).peel_refs();

                if_chain! {
                    if (path.ident.name == sym!(mode)
                        && (match_type(cx, obj_ty, &paths::OPEN_OPTIONS)
                            || match_type(cx, obj_ty, &paths::DIR_BUILDER)))
                        || (path.ident.name == sym!(set_mode) && match_type(cx, obj_ty, &paths::PERMISSIONS));
                    if let ExprKind::Lit(_) = param.kind;

                    then {
                        let snip = match snippet_opt(cx, param.span) {
                            Some(s) => s,
                            _ => return,
                        };

                        if !snip.starts_with("0o") {
                            show_error(cx, param);
                        }
                    }
                }
            },
            ExprKind::Call(func, [param]) => {
                if_chain! {
                    if let ExprKind::Path(ref path) = func.kind;
                    if let Some(def_id) = cx.qpath_res(path, func.hir_id).opt_def_id();
                    if match_def_path(cx, def_id, &paths::PERMISSIONS_FROM_MODE);
                    if let ExprKind::Lit(_) = param.kind;

                    then {
                        let snip = match snippet_opt(cx, param.span) {
                            Some(s) => s,
                            _ => return,
                        };

                        if !snip.starts_with("0o") {
                            show_error(cx, param);
                        }
                    }
                }
            },
            _ => {},
        };
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
