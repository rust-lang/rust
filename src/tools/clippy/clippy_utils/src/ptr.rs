use crate::source::snippet;
use crate::visitors::expr_visitor_no_bodies;
use crate::{path_to_local_id, strip_pat_refs};
use rustc_hir::intravisit::Visitor;
use rustc_hir::{Body, BodyId, ExprKind, HirId, PatKind};
use rustc_lint::LateContext;
use rustc_span::Span;
use std::borrow::Cow;

pub fn get_spans(
    cx: &LateContext<'_>,
    opt_body_id: Option<BodyId>,
    idx: usize,
    replacements: &[(&'static str, &'static str)],
) -> Option<Vec<(Span, Cow<'static, str>)>> {
    if let Some(body) = opt_body_id.map(|id| cx.tcx.hir().body(id)) {
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
    replace: &[(&'static str, &'static str)],
    body: &'tcx Body<'_>,
) -> Option<Vec<(Span, Cow<'static, str>)>> {
    let mut abort = false;
    let mut spans = Vec::new();
    expr_visitor_no_bodies(|expr| {
        if abort {
            return false;
        }
        if let ExprKind::MethodCall(seg, [recv], _) = expr.kind {
            if path_to_local_id(recv, id) {
                if seg.ident.name.as_str() == "capacity" {
                    abort = true;
                    return false;
                }
                for &(fn_name, suffix) in replace {
                    if seg.ident.name.as_str() == fn_name {
                        spans.push((expr.span, snippet(cx, recv.span, "_") + suffix));
                        return false;
                    }
                }
            }
        }
        !abort
    })
    .visit_body(body);
    if abort { None } else { Some(spans) }
}
