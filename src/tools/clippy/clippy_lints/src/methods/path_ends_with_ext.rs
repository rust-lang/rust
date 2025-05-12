use super::PATH_ENDS_WITH_EXT;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_ast::{LitKind, StrStyle};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::sym;
use std::fmt::Write;

pub const DEFAULT_ALLOWED_DOTFILES: &[&str] = &[
    "git", "svn", "gem", "npm", "vim", "env", "rnd", "ssh", "vnc", "smb", "nvm", "bin",
];

pub(super) fn check(
    cx: &LateContext<'_>,
    recv: &Expr<'_>,
    path: &Expr<'_>,
    expr: &Expr<'_>,
    msrv: Msrv,
    allowed_dotfiles: &FxHashSet<&'static str>,
) {
    if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv).peel_refs(), sym::Path)
        && !path.span.from_expansion()
        && let ExprKind::Lit(lit) = path.kind
        && let LitKind::Str(path, StrStyle::Cooked) = lit.node
        && let Some(path) = path.as_str().strip_prefix('.')
        && (1..=3).contains(&path.len())
        && !allowed_dotfiles.contains(path)
        && path.chars().all(char::is_alphanumeric)
    {
        let mut sugg = snippet(cx, recv.span, "..").into_owned();
        if msrv.meets(cx, msrvs::OPTION_RESULT_IS_VARIANT_AND) {
            let _ = write!(sugg, r#".extension().is_some_and(|ext| ext == "{path}")"#);
        } else {
            let _ = write!(sugg, r#".extension().map_or(false, |ext| ext == "{path}")"#);
        }

        span_lint_and_sugg(
            cx,
            PATH_ENDS_WITH_EXT,
            expr.span,
            "this looks like a failed attempt at checking for the file extension",
            "try",
            sugg,
            Applicability::MaybeIncorrect,
        );
    }
}
