use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;
use std::path::{Component, Path};

use super::PATH_BUF_PUSH_OVERWRITE;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, arg: &'tcx Expr<'_>) {
    if_chain! {
        if let Some(method_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id);
        if let Some(impl_id) = cx.tcx.impl_of_method(method_id);
        if is_type_diagnostic_item(cx, cx.tcx.type_of(impl_id).subst_identity(), sym::PathBuf);
        if let ExprKind::Lit(ref lit) = arg.kind;
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
