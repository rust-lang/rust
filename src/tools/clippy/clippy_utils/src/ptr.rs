use crate::source::snippet;
use crate::visitors::{Descend, for_each_expr_without_closures};
use crate::{path_to_local_id, strip_pat_refs, sym};
use core::ops::ControlFlow;
use rustc_hir::{Body, BodyId, ExprKind, HirId, PatKind};
use rustc_lint::LateContext;
use rustc_span::{Span, Symbol};
use std::borrow::Cow;

pub fn get_spans(
    cx: &LateContext<'_>,
    opt_body_id: Option<BodyId>,
    idx: usize,
    replacements: &[(Symbol, &'static str)],
) -> Option<Vec<(Span, Cow<'static, str>)>> {
    if let Some(body) = opt_body_id.map(|id| cx.tcx.hir_body(id)) {
        if let PatKind::Binding(_, binding_id, _, _) = strip_pat_refs(body.params[idx].pat).kind {
            extract_clone_suggestions(cx, binding_id, replacements, body)
        } else {
            Some(vec![])
        }
    } else {
        Some(vec![])
    }
}

fn extract_clone_suggestions<'tcx>(
    cx: &LateContext<'tcx>,
    id: HirId,
    replace: &[(Symbol, &'static str)],
    body: &'tcx Body<'_>,
) -> Option<Vec<(Span, Cow<'static, str>)>> {
    let mut spans = Vec::new();
    for_each_expr_without_closures(body, |e| {
        if let ExprKind::MethodCall(seg, recv, [], _) = e.kind
            && path_to_local_id(recv, id)
        {
            if seg.ident.name == sym::capacity {
                return ControlFlow::Break(());
            }
            for &(fn_name, suffix) in replace {
                if seg.ident.name == fn_name {
                    spans.push((e.span, snippet(cx, recv.span, "_") + suffix));
                    return ControlFlow::Continue(Descend::No);
                }
            }
        }
        ControlFlow::Continue(Descend::Yes)
    })
    .is_none()
    .then_some(spans)
}
