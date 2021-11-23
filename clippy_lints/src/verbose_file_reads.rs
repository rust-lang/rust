use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::paths;
use clippy_utils::ty::match_type;
use if_chain::if_chain;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for use of File::read_to_end and File::read_to_string.
    ///
    /// ### Why is this bad?
    /// `fs::{read, read_to_string}` provide the same functionality when `buf` is empty with fewer imports and no intermediate values.
    /// See also: [fs::read docs](https://doc.rust-lang.org/std/fs/fn.read.html), [fs::read_to_string docs](https://doc.rust-lang.org/std/fs/fn.read_to_string.html)
    ///
    /// ### Example
    /// ```rust,no_run
    /// # use std::io::Read;
    /// # use std::fs::File;
    /// let mut f = File::open("foo.txt").unwrap();
    /// let mut bytes = Vec::new();
    /// f.read_to_end(&mut bytes).unwrap();
    /// ```
    /// Can be written more concisely as
    /// ```rust,no_run
    /// # use std::fs;
    /// let mut bytes = fs::read("foo.txt").unwrap();
    /// ```
    #[clippy::version = "1.44.0"]
    pub VERBOSE_FILE_READS,
    restriction,
    "use of `File::read_to_end` or `File::read_to_string`"
}

declare_lint_pass!(VerboseFileReads => [VERBOSE_FILE_READS]);

impl<'tcx> LateLintPass<'tcx> for VerboseFileReads {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if is_file_read_to_end(cx, expr) {
            span_lint_and_help(
                cx,
                VERBOSE_FILE_READS,
                expr.span,
                "use of `File::read_to_end`",
                None,
                "consider using `fs::read` instead",
            );
        } else if is_file_read_to_string(cx, expr) {
            span_lint_and_help(
                cx,
                VERBOSE_FILE_READS,
                expr.span,
                "use of `File::read_to_string`",
                None,
                "consider using `fs::read_to_string` instead",
            );
        }
    }
}

fn is_file_read_to_end<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    if_chain! {
        if let ExprKind::MethodCall(method_name, _, exprs, _) = expr.kind;
        if method_name.ident.as_str() == "read_to_end";
        if let ExprKind::Path(QPath::Resolved(None, _)) = &exprs[0].kind;
        let ty = cx.typeck_results().expr_ty(&exprs[0]);
        if match_type(cx, ty, &paths::FILE);
        then {
            return true
        }
    }
    false
}

fn is_file_read_to_string<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    if_chain! {
        if let ExprKind::MethodCall(method_name, _, exprs, _) = expr.kind;
        if method_name.ident.as_str() == "read_to_string";
        if let ExprKind::Path(QPath::Resolved(None, _)) = &exprs[0].kind;
        let ty = cx.typeck_results().expr_ty(&exprs[0]);
        if match_type(cx, ty, &paths::FILE);
        then {
            return true
        }
    }
    false
}
