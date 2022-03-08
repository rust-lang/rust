use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::{get_iterator_item_ty, is_copy};
use itertools::Itertools;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_middle::ty;
use std::ops::Not;

use super::ITER_OVEREAGER_CLONED;
use crate::redundant_clone::REDUNDANT_CLONE;

/// lint overeager use of `cloned()` for `Iterator`s
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    recv: &'tcx hir::Expr<'_>,
    name: &str,
    map_arg: &[hir::Expr<'_>],
) {
    // Check if it's iterator and get type associated with `Item`.
    let inner_ty = match get_iterator_item_ty(cx, cx.typeck_results().expr_ty_adjusted(recv)) {
        Some(ty) => ty,
        _ => return,
    };

    match inner_ty.kind() {
        ty::Ref(_, ty, _) if !is_copy(cx, *ty) => {},
        _ => return,
    };

    let (lint, preserve_cloned) = match name {
        "count" => (REDUNDANT_CLONE, false),
        _ => (ITER_OVEREAGER_CLONED, true),
    };
    let wildcard_params = map_arg.is_empty().not().then(|| "...").unwrap_or_default();
    let msg = format!(
        "called `cloned().{}({})` on an `Iterator`. It may be more efficient to call `{}({}){}` instead",
        name,
        wildcard_params,
        name,
        wildcard_params,
        preserve_cloned.then(|| ".cloned()").unwrap_or_default(),
    );

    span_lint_and_sugg(
        cx,
        lint,
        expr.span,
        &msg,
        "try this",
        format!(
            "{}.{}({}){}",
            snippet(cx, recv.span, ".."),
            name,
            map_arg.iter().map(|a| snippet(cx, a.span, "..")).join(", "),
            preserve_cloned.then(|| ".cloned()").unwrap_or_default(),
        ),
        Applicability::MachineApplicable,
    );
}
