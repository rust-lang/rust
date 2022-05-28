use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::sym;
use std::path::{Component, Path};

declare_clippy_lint! {
    /// ### What it does
    ///* Checks for [push](https://doc.rust-lang.org/std/path/struct.PathBuf.html#method.push)
    /// calls on `PathBuf` that can cause overwrites.
    ///
    /// ### Why is this bad?
    /// Calling `push` with a root path at the start can overwrite the
    /// previous defined path.
    ///
    /// ### Example
    /// ```rust
    /// use std::path::PathBuf;
    ///
    /// let mut x = PathBuf::from("/foo");
    /// x.push("/bar");
    /// assert_eq!(x, PathBuf::from("/bar"));
    /// ```
    /// Could be written:
    ///
    /// ```rust
    /// use std::path::PathBuf;
    ///
    /// let mut x = PathBuf::from("/foo");
    /// x.push("bar");
    /// assert_eq!(x, PathBuf::from("/foo/bar"));
    /// ```
    #[clippy::version = "1.36.0"]
    pub PATH_BUF_PUSH_OVERWRITE,
    nursery,
    "calling `push` with file system root on `PathBuf` can overwrite it"
}

declare_lint_pass!(PathBufPushOverwrite => [PATH_BUF_PUSH_OVERWRITE]);

impl<'tcx> LateLintPass<'tcx> for PathBufPushOverwrite {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::MethodCall(path, args, _) = expr.kind;
            if path.ident.name == sym!(push);
            if args.len() == 2;
            if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(&args[0]).peel_refs(), sym::PathBuf);
            if let Some(get_index_arg) = args.get(1);
            if let ExprKind::Lit(ref lit) = get_index_arg.kind;
            if let LitKind::Str(ref path_lit, _) = lit.node;
            if let pushed_path = Path::new(path_lit.as_str());
            if let Some(pushed_path_lit) = pushed_path.to_str();
            if pushed_path.has_root();
            if let Some(root) = pushed_path.components().next();
            if root == Component::RootDir;
            then {
                span_lint_and_sugg(
                    cx,
                    PATH_BUF_PUSH_OVERWRITE,
                    lit.span,
                    "calling `push` with '/' or '\\' (file system root) will overwrite the previous path definition",
                    "try",
                    format!("\"{}\"", pushed_path_lit.trim_start_matches(|c| c == '/' || c == '\\')),
                    Applicability::MachineApplicable,
                );
            }
        }
    }
}
