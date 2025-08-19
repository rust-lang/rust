use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;
use std::path::{Component, Path};

use super::PATH_BUF_PUSH_OVERWRITE;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, arg: &'tcx Expr<'_>) {
    if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
        && let Some(impl_id) = cx.tcx.impl_of_assoc(method_id)
        && is_type_diagnostic_item(cx, cx.tcx.type_of(impl_id).instantiate_identity(), sym::PathBuf)
        && let ExprKind::Lit(lit) = arg.kind
        && let LitKind::Str(ref path_lit, _) = lit.node
        && let pushed_path = Path::new(path_lit.as_str())
        && let Some(pushed_path_lit) = pushed_path.to_str()
        && pushed_path.has_root()
        && let Some(root) = pushed_path.components().next()
        && root == Component::RootDir
    {
        span_lint_and_sugg(
            cx,
            PATH_BUF_PUSH_OVERWRITE,
            lit.span,
            "calling `push` with '/' or '\\' (file system root) will overwrite the previous path definition",
            "try",
            format!("\"{}\"", pushed_path_lit.trim_start_matches(['/', '\\'])),
            Applicability::MaybeIncorrect,
        );
    }
}
