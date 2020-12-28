use crate::utils::{get_pat_name, match_var, snippet};
use rustc_hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{Body, BodyId, Expr, ExprKind, Param};
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;
use rustc_span::{Span, Symbol};
use std::borrow::Cow;

pub fn get_spans(
    cx: &LateContext<'_>,
    opt_body_id: Option<BodyId>,
    idx: usize,
    replacements: &[(&'static str, &'static str)],
) -> Option<Vec<(Span, Cow<'static, str>)>> {
    if let Some(body) = opt_body_id.map(|id| cx.tcx.hir().body(id)) {
        get_binding_name(&body.params[idx]).map_or_else(
            || Some(vec![]),
            |name| extract_clone_suggestions(cx, name, replacements, body),
        )
    } else {
        Some(vec![])
    }
}

fn extract_clone_suggestions<'tcx>(
    cx: &LateContext<'tcx>,
    name: Symbol,
    replace: &[(&'static str, &'static str)],
    body: &'tcx Body<'_>,
) -> Option<Vec<(Span, Cow<'static, str>)>> {
    let mut visitor = PtrCloneVisitor {
        cx,
        name,
        replace,
        spans: vec![],
        abort: false,
    };
    visitor.visit_body(body);
    if visitor.abort {
        None
    } else {
        Some(visitor.spans)
    }
}

struct PtrCloneVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    name: Symbol,
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
        if let ExprKind::MethodCall(ref seg, _, ref args, _) = expr.kind {
            if args.len() == 1 && match_var(&args[0], self.name) {
                if seg.ident.name.as_str() == "capacity" {
                    self.abort = true;
                    return;
                }
                for &(fn_name, suffix) in self.replace {
                    if seg.ident.name.as_str() == fn_name {
                        self.spans
                            .push((expr.span, snippet(self.cx, args[0].span, "_") + suffix));
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

fn get_binding_name(arg: &Param<'_>) -> Option<Symbol> {
    get_pat_name(&arg.pat)
}
