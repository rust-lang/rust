use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::is_copy;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_path, Visitor};
use rustc_hir::{self, HirId, Path};
use rustc_lint::LateContext;
use rustc_middle::hir::nested_filter;
use rustc_span::source_map::Span;
use rustc_span::{sym, Symbol};

use super::MAP_UNWRAP_OR;

/// lint use of `map().unwrap_or()` for `Option`s
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &rustc_hir::Expr<'_>,
    recv: &rustc_hir::Expr<'_>,
    map_arg: &'tcx rustc_hir::Expr<'_>,
    unwrap_recv: &rustc_hir::Expr<'_>,
    unwrap_arg: &'tcx rustc_hir::Expr<'_>,
    map_span: Span,
) {
    // lint if the caller of `map()` is an `Option`
    if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Option) {
        if !is_copy(cx, cx.typeck_results().expr_ty(unwrap_arg)) {
            // Do not lint if the `map` argument uses identifiers in the `map`
            // argument that are also used in the `unwrap_or` argument

            let mut unwrap_visitor = UnwrapVisitor {
                cx,
                identifiers: FxHashSet::default(),
            };
            unwrap_visitor.visit_expr(unwrap_arg);

            let mut map_expr_visitor = MapExprVisitor {
                cx,
                identifiers: unwrap_visitor.identifiers,
                found_identifier: false,
            };
            map_expr_visitor.visit_expr(map_arg);

            if map_expr_visitor.found_identifier {
                return;
            }
        }

        if unwrap_arg.span.ctxt() != map_span.ctxt() {
            return;
        }

        let mut applicability = Applicability::MachineApplicable;
        // get snippet for unwrap_or()
        let unwrap_snippet = snippet_with_applicability(cx, unwrap_arg.span, "..", &mut applicability);
        // lint message
        // comparing the snippet from source to raw text ("None") below is safe
        // because we already have checked the type.
        let arg = if unwrap_snippet == "None" { "None" } else { "<a>" };
        let unwrap_snippet_none = unwrap_snippet == "None";
        let suggest = if unwrap_snippet_none {
            "and_then(<f>)"
        } else {
            "map_or(<a>, <f>)"
        };
        let msg = &format!(
            "called `map(<f>).unwrap_or({arg})` on an `Option` value. \
            This can be done more directly by calling `{suggest}` instead"
        );

        span_lint_and_then(cx, MAP_UNWRAP_OR, expr.span, msg, |diag| {
            let map_arg_span = map_arg.span;

            let mut suggestion = vec![
                (
                    map_span,
                    String::from(if unwrap_snippet_none { "and_then" } else { "map_or" }),
                ),
                (expr.span.with_lo(unwrap_recv.span.hi()), String::new()),
            ];

            if !unwrap_snippet_none {
                suggestion.push((map_arg_span.with_hi(map_arg_span.lo()), format!("{unwrap_snippet}, ")));
            }

            diag.multipart_suggestion(&format!("use `{suggest}` instead"), suggestion, applicability);
        });
    }
}

struct UnwrapVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    identifiers: FxHashSet<Symbol>,
}

impl<'a, 'tcx> Visitor<'tcx> for UnwrapVisitor<'a, 'tcx> {
    type NestedFilter = nested_filter::All;

    fn visit_path(&mut self, path: &Path<'tcx>, _id: HirId) {
        self.identifiers.insert(ident(path));
        walk_path(self, path);
    }

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }
}

struct MapExprVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    identifiers: FxHashSet<Symbol>,
    found_identifier: bool,
}

impl<'a, 'tcx> Visitor<'tcx> for MapExprVisitor<'a, 'tcx> {
    type NestedFilter = nested_filter::All;

    fn visit_path(&mut self, path: &Path<'tcx>, _id: HirId) {
        if self.identifiers.contains(&ident(path)) {
            self.found_identifier = true;
            return;
        }
        walk_path(self, path);
    }

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }
}

fn ident(path: &Path<'_>) -> Symbol {
    path.segments
        .last()
        .expect("segments should be composed of at least 1 element")
        .ident
        .name
}
