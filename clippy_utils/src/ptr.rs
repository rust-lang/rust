use crate::source::snippet;
use crate::{path_to_local_id, strip_pat_refs};
use rustc_hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{Body, BodyId, Expr, ExprKind, HirId, PatKind};
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;
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
    let mut visitor = PtrCloneVisitor {
        cx,
        id,
        replace,
        spans: vec![],
        abort: false,
    };
    visitor.visit_body(body);
    if visitor.abort { None } else { Some(visitor.spans) }
}

struct PtrCloneVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    id: HirId,
    replace: &'a [(&'static str, &'static str)],
    spans: Vec<(Span, Cow<'static, str>)>,
    abort: bool,
}

impl<'a, 'tcx> Visitor<'tcx> for PtrCloneVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if self.abort {
            return;
        }
        if let ExprKind::MethodCall(seg, _, [recv], _) = expr.kind {
            if path_to_local_id(recv, self.id) {
                if seg.ident.name.as_str() == "capacity" {
                    self.abort = true;
                    return;
                }
                for &(fn_name, suffix) in self.replace {
                    if seg.ident.name.as_str() == fn_name {
                        self.spans.push((expr.span, snippet(self.cx, recv.span, "_") + suffix));
                        return;
                    }
                }
            }
        }
        walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}
