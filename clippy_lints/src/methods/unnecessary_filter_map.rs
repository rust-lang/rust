use crate::utils::paths;
use crate::utils::usage::mutated_variables;
use crate::utils::{match_qpath, match_trait_method, span_lint};
use rustc::hir;
use rustc::hir::def::Res;
use rustc::hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use rustc::lint::LateContext;

use if_chain::if_chain;

use super::UNNECESSARY_FILTER_MAP;

pub(super) fn lint(cx: &LateContext<'_, '_>, expr: &hir::Expr, args: &[hir::Expr]) {
    if !match_trait_method(cx, expr, &paths::ITERATOR) {
        return;
    }

    if let hir::ExprKind::Closure(_, _, body_id, ..) = args[1].node {
        let body = cx.tcx.hir().body(body_id);
        let arg_id = body.arguments[0].pat.hir_id;
        let mutates_arg = match mutated_variables(&body.value, cx) {
            Some(used_mutably) => used_mutably.contains(&arg_id),
            None => true,
        };

        let (mut found_mapping, mut found_filtering) = check_expression(&cx, arg_id, &body.value);

        let mut return_visitor = ReturnVisitor::new(&cx, arg_id);
        return_visitor.visit_expr(&body.value);
        found_mapping |= return_visitor.found_mapping;
        found_filtering |= return_visitor.found_filtering;

        if !found_filtering {
            span_lint(
                cx,
                UNNECESSARY_FILTER_MAP,
                expr.span,
                "this `.filter_map` can be written more simply using `.map`",
            );
            return;
        }

        if !found_mapping && !mutates_arg {
            span_lint(
                cx,
                UNNECESSARY_FILTER_MAP,
                expr.span,
                "this `.filter_map` can be written more simply using `.filter`",
            );
            return;
        }
    }
}

// returns (found_mapping, found_filtering)
fn check_expression<'a, 'tcx>(
    cx: &'a LateContext<'a, 'tcx>,
    arg_id: hir::HirId,
    expr: &'tcx hir::Expr,
) -> (bool, bool) {
    match &expr.node {
        hir::ExprKind::Call(ref func, ref args) => {
            if_chain! {
                if let hir::ExprKind::Path(ref path) = func.node;
                then {
                    if match_qpath(path, &paths::OPTION_SOME) {
                        if_chain! {
                            if let hir::ExprKind::Path(path) = &args[0].node;
                            if let Res::Local(ref local) = cx.tables.qpath_res(path, args[0].hir_id);
                            then {
                                if arg_id == *local {
                                    return (false, false)
                                }
                            }
                        }
                        return (true, false);
                    } else {
                        // We don't know. It might do anything.
                        return (true, true);
                    }
                }
            }
            (true, true)
        },
        hir::ExprKind::Block(ref block, _) => {
            if let Some(expr) = &block.expr {
                check_expression(cx, arg_id, &expr)
            } else {
                (false, false)
            }
        },
        hir::ExprKind::Match(_, ref arms, _) => {
            let mut found_mapping = false;
            let mut found_filtering = false;
            for arm in arms {
                let (m, f) = check_expression(cx, arg_id, &arm.body);
                found_mapping |= m;
                found_filtering |= f;
            }
            (found_mapping, found_filtering)
        },
        hir::ExprKind::Path(path) if match_qpath(path, &paths::OPTION_NONE) => (false, true),
        _ => (true, true),
    }
}

struct ReturnVisitor<'a, 'tcx> {
    cx: &'a LateContext<'a, 'tcx>,
    arg_id: hir::HirId,
    // Found a non-None return that isn't Some(input)
    found_mapping: bool,
    // Found a return that isn't Some
    found_filtering: bool,
}

impl<'a, 'tcx> ReturnVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'a, 'tcx>, arg_id: hir::HirId) -> ReturnVisitor<'a, 'tcx> {
        ReturnVisitor {
            cx,
            arg_id,
            found_mapping: false,
            found_filtering: false,
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for ReturnVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr) {
        if let hir::ExprKind::Ret(Some(expr)) = &expr.node {
            let (found_mapping, found_filtering) = check_expression(self.cx, self.arg_id, expr);
            self.found_mapping |= found_mapping;
            self.found_filtering |= found_filtering;
        } else {
            walk_expr(self, expr);
        }
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}
