use crate::utils::paths;
use crate::utils::{is_copy, match_type, snippet, span_lint, span_note_and_lint};
use rustc::hir::intravisit::{walk_path, NestedVisitorMap, Visitor};
use rustc::hir::{self, *};
use rustc::lint::LateContext;
use rustc_data_structures::fx::FxHashSet;
use syntax_pos::symbol::Symbol;

use super::OPTION_MAP_UNWRAP_OR;

/// lint use of `map().unwrap_or()` for `Option`s
pub(super) fn lint<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    expr: &hir::Expr,
    map_args: &'tcx [hir::Expr],
    unwrap_args: &'tcx [hir::Expr],
) {
    // lint if the caller of `map()` is an `Option`
    if match_type(cx, cx.tables.expr_ty(&map_args[0]), &paths::OPTION) {
        if !is_copy(cx, cx.tables.expr_ty(&unwrap_args[1])) {
            // Do not lint if the `map` argument uses identifiers in the `map`
            // argument that are also used in the `unwrap_or` argument

            let mut unwrap_visitor = UnwrapVisitor {
                cx,
                identifiers: FxHashSet::default(),
            };
            unwrap_visitor.visit_expr(&unwrap_args[1]);

            let mut map_expr_visitor = MapExprVisitor {
                cx,
                identifiers: unwrap_visitor.identifiers,
                found_identifier: false,
            };
            map_expr_visitor.visit_expr(&map_args[1]);

            if map_expr_visitor.found_identifier {
                return;
            }
        }

        // get snippets for args to map() and unwrap_or()
        let map_snippet = snippet(cx, map_args[1].span, "..");
        let unwrap_snippet = snippet(cx, unwrap_args[1].span, "..");
        // lint message
        // comparing the snippet from source to raw text ("None") below is safe
        // because we already have checked the type.
        let arg = if unwrap_snippet == "None" { "None" } else { "a" };
        let suggest = if unwrap_snippet == "None" {
            "and_then(f)"
        } else {
            "map_or(a, f)"
        };
        let msg = &format!(
            "called `map(f).unwrap_or({})` on an Option value. \
             This can be done more directly by calling `{}` instead",
            arg, suggest
        );
        // lint, with note if neither arg is > 1 line and both map() and
        // unwrap_or() have the same span
        let multiline = map_snippet.lines().count() > 1 || unwrap_snippet.lines().count() > 1;
        let same_span = map_args[1].span.ctxt() == unwrap_args[1].span.ctxt();
        if same_span && !multiline {
            let suggest = if unwrap_snippet == "None" {
                format!("and_then({})", map_snippet)
            } else {
                format!("map_or({}, {})", unwrap_snippet, map_snippet)
            };
            let note = format!(
                "replace `map({}).unwrap_or({})` with `{}`",
                map_snippet, unwrap_snippet, suggest
            );
            span_note_and_lint(cx, OPTION_MAP_UNWRAP_OR, expr.span, msg, expr.span, &note);
        } else if same_span && multiline {
            span_lint(cx, OPTION_MAP_UNWRAP_OR, expr.span, msg);
        };
    }
}

struct UnwrapVisitor<'a, 'tcx> {
    cx: &'a LateContext<'a, 'tcx>,
    identifiers: FxHashSet<Symbol>,
}

impl<'a, 'tcx> Visitor<'tcx> for UnwrapVisitor<'a, 'tcx> {
    fn visit_path(&mut self, path: &'tcx Path, _id: HirId) {
        self.identifiers.insert(ident(path));
        walk_path(self, path);
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.cx.tcx.hir())
    }
}

struct MapExprVisitor<'a, 'tcx> {
    cx: &'a LateContext<'a, 'tcx>,
    identifiers: FxHashSet<Symbol>,
    found_identifier: bool,
}

impl<'a, 'tcx> Visitor<'tcx> for MapExprVisitor<'a, 'tcx> {
    fn visit_path(&mut self, path: &'tcx Path, _id: HirId) {
        if self.identifiers.contains(&ident(path)) {
            self.found_identifier = true;
            return;
        }
        walk_path(self, path);
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.cx.tcx.hir())
    }
}

fn ident(path: &Path) -> Symbol {
    path.segments
        .last()
        .expect("segments should be composed of at least 1 element")
        .ident
        .name
}
