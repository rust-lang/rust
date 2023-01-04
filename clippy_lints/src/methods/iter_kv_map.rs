#![allow(unused_imports)]

use super::ITER_KV_MAP;
use clippy_utils::diagnostics::{multispan_sugg, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::sugg;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::visitors::is_local_used;
use rustc_hir::{BindingAnnotation, Body, BorrowKind, ByRef, Expr, ExprKind, Mutability, Pat, PatKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::ty;
use rustc_span::sym;
use rustc_span::Span;

/// lint use of:
/// - `hashmap.iter().map(|(_, v)| v)`
/// - `hashmap.into_iter().map(|(_, v)| v)`
/// on `HashMaps` and `BTreeMaps` in std

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    map_type: &'tcx str,     // iter / into_iter
    expr: &'tcx Expr<'tcx>,  // .iter().map(|(_, v_| v))
    recv: &'tcx Expr<'tcx>,  // hashmap
    m_arg: &'tcx Expr<'tcx>, // |(_, v)| v
) {
    if_chain! {
        if !expr.span.from_expansion();
        if let ExprKind::Closure(c) = m_arg.kind;
        if let Body {params: [p], value: body_expr, generator_kind: _ } = cx.tcx.hir().body(c.body);
        if let PatKind::Tuple([key_pat, val_pat], _) = p.pat.kind;

        let (replacement_kind, annotation, binded_ident) = match (&key_pat.kind, &val_pat.kind) {
            (key, PatKind::Binding(ann, _, value, _)) if pat_is_wild(cx, key, m_arg) => ("value", ann, value),
            (PatKind::Binding(ann, _, key, _), value) if pat_is_wild(cx, value, m_arg) => ("key", ann, key),
            _ => return,
        };

        let ty = cx.typeck_results().expr_ty(recv);
        if is_type_diagnostic_item(cx, ty, sym::HashMap) || is_type_diagnostic_item(cx, ty, sym::BTreeMap);

        then {
            let mut applicability = rustc_errors::Applicability::MachineApplicable;
            let recv_snippet = snippet_with_applicability(cx, recv.span, "map", &mut applicability);
            let into_prefix = if map_type == "into_iter" {"into_"} else {""};

            if_chain! {
                if let ExprKind::Path(rustc_hir::QPath::Resolved(_, path)) = body_expr.kind;
                if let [local_ident] = path.segments;
                if local_ident.ident.as_str() == binded_ident.as_str();

                then {
                    span_lint_and_sugg(
                        cx,
                        ITER_KV_MAP,
                        expr.span,
                        &format!("iterating on a map's {replacement_kind}s"),
                        "try",
                        format!("{recv_snippet}.{into_prefix}{replacement_kind}s()"),
                        applicability,
                    );
                } else {
                    let ref_annotation = if annotation.0 == ByRef::Yes {
                        "ref "
                    } else {
                        ""
                    };
                    let mut_annotation = if annotation.1 == Mutability::Mut {
                        "mut "
                    } else {
                        ""
                    };
                    span_lint_and_sugg(
                        cx,
                        ITER_KV_MAP,
                        expr.span,
                        &format!("iterating on a map's {replacement_kind}s"),
                        "try",
                        format!("{recv_snippet}.{into_prefix}{replacement_kind}s().map(|{ref_annotation}{mut_annotation}{binded_ident}| {})",
                            snippet_with_applicability(cx, body_expr.span, "/* body */", &mut applicability)),
                        applicability,
                    );
                }
            }
        }
    }
}

/// Returns `true` if the pattern is a `PatWild`, or is an ident prefixed with `_`
/// that is not locally used.
fn pat_is_wild<'tcx>(cx: &LateContext<'tcx>, pat: &'tcx PatKind<'_>, body: &'tcx Expr<'_>) -> bool {
    match *pat {
        PatKind::Wild => true,
        PatKind::Binding(_, id, ident, None) if ident.as_str().starts_with('_') => !is_local_used(cx, body, id),
        _ => false,
    }
}
