use rustc_errors::Applicability;
use rustc_hir::{
    intravisit::{walk_expr, NestedVisitorMap, Visitor},
    Expr, ExprKind,
};
use rustc_lint::LateContext;
use rustc_middle::{hir::map::Map, ty, ty::Ty};
use rustc_span::source_map::Span;

use crate::utils::{match_trait_method, match_type, paths, snippet, span_lint_and_then};

use if_chain::if_chain;

pub(super) fn lint(cx: &LateContext<'_>, expr: &'tcx Expr<'_>, args: &[&[Expr<'_>]]) {
    if args.len() < 2 {
        return;
    }

    let for_each_args = args[0];
    if for_each_args.len() < 2 {
        return;
    }
    let for_each_receiver = &for_each_args[0];
    let for_each_arg = &for_each_args[1];
    let iter_receiver = &args[1][0];

    if_chain! {
        if match_trait_method(cx, expr, &paths::ITERATOR);
        if is_target_ty(cx, cx.typeck_results().expr_ty(iter_receiver));
        if let ExprKind::Closure(_, _, body_id, ..) = for_each_arg.kind;
        let body = cx.tcx.hir().body(body_id);
        if let ExprKind::Block(..) = body.value.kind;
        then {
            let mut ret_span_collector = RetSpanCollector::new();
            ret_span_collector.visit_expr(&body.value);

            let label = "'outer";
            let loop_label = if ret_span_collector.need_label {
                format!("{}: ", label)
            } else {
                "".to_string()
            };
            let sugg =
                format!("{}for {} in {} {{ .. }}", loop_label, snippet(cx, body.params[0].pat.span, ""), snippet(cx, for_each_receiver.span, ""));

            let mut notes = vec![];
            for (span, need_label) in ret_span_collector.spans {
                let cont_label = if need_label {
                    format!(" {}", label)
                } else {
                    "".to_string()
                };
                let note = format!("change `return` to `continue{}` in the loop body", cont_label);
                notes.push((span, note));
            }

            span_lint_and_then(cx,
                      super::EXCESSIVE_FOR_EACH,
                      expr.span,
                      "excessive use of `for_each`",
                      |diag| {
                          diag.span_suggestion(expr.span, "try", sugg, Applicability::HasPlaceholders);
                          for note in notes {
                              diag.span_note(note.0, &note.1);
                          }
                      }
                );
        }
    }
}

type PathSegment = &'static [&'static str];

const TARGET_ITER_RECEIVER_TY: &[PathSegment] = &[
    &paths::VEC,
    &paths::VEC_DEQUE,
    &paths::LINKED_LIST,
    &paths::HASHMAP,
    &paths::BTREEMAP,
    &paths::HASHSET,
    &paths::BTREESET,
    &paths::BINARY_HEAP,
];

fn is_target_ty(cx: &LateContext<'_>, expr_ty: Ty<'_>) -> bool {
    let expr_ty = expr_ty.peel_refs();
    for target in TARGET_ITER_RECEIVER_TY {
        if match_type(cx, expr_ty, target) {
            return true;
        }
    }

    if_chain! {
        if matches!(expr_ty.kind(), ty::Slice(_) | ty::Array(..));
        then {
            return true;
        }
    }

    false
}

/// Collect spans of `return` in the closure body.
struct RetSpanCollector {
    spans: Vec<(Span, bool)>,
    loop_depth: u16,
    need_label: bool,
}

impl RetSpanCollector {
    fn new() -> Self {
        Self {
            spans: Vec::new(),
            loop_depth: 0,
            need_label: false,
        }
    }
}

impl<'tcx> Visitor<'tcx> for RetSpanCollector {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &Expr<'_>) {
        match expr.kind {
            ExprKind::Ret(..) => {
                if self.loop_depth > 0 && !self.need_label {
                    self.need_label = true
                }

                self.spans.push((expr.span, self.loop_depth > 0))
            },

            ExprKind::Loop(..) => {
                self.loop_depth += 1;
                walk_expr(self, expr);
                self.loop_depth -= 1;
                return;
            },

            _ => {},
        }

        walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}
