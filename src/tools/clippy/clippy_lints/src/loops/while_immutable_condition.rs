use super::WHILE_IMMUTABLE_CONDITION;
use clippy_utils::consts::constant;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::usage::mutated_variables;
use if_chain::if_chain;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefIdMap;
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::{Expr, ExprKind, HirIdSet, QPath};
use rustc_lint::LateContext;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, cond: &'tcx Expr<'_>, expr: &'tcx Expr<'_>) {
    if constant(cx, cx.typeck_results(), cond).is_some() {
        // A pure constant condition (e.g., `while false`) is not linted.
        return;
    }

    let mut var_visitor = VarCollectorVisitor {
        cx,
        ids: HirIdSet::default(),
        def_ids: DefIdMap::default(),
        skip: false,
    };
    var_visitor.visit_expr(cond);
    if var_visitor.skip {
        return;
    }
    let used_in_condition = &var_visitor.ids;
    let mutated_in_body = mutated_variables(expr, cx);
    let mutated_in_condition = mutated_variables(cond, cx);
    let no_cond_variable_mutated =
        if let (Some(used_mutably_body), Some(used_mutably_cond)) = (mutated_in_body, mutated_in_condition) {
            used_in_condition.is_disjoint(&used_mutably_body) && used_in_condition.is_disjoint(&used_mutably_cond)
        } else {
            return;
        };
    let mutable_static_in_cond = var_visitor.def_ids.items().any(|(_, v)| *v);

    let mut has_break_or_return_visitor = HasBreakOrReturnVisitor {
        has_break_or_return: false,
    };
    has_break_or_return_visitor.visit_expr(expr);
    let has_break_or_return = has_break_or_return_visitor.has_break_or_return;

    if no_cond_variable_mutated && !mutable_static_in_cond {
        span_lint_and_then(
            cx,
            WHILE_IMMUTABLE_CONDITION,
            cond.span,
            "variables in the condition are not mutated in the loop body",
            |diag| {
                diag.note("this may lead to an infinite or to a never running loop");

                if has_break_or_return {
                    diag.note("this loop contains `return`s or `break`s");
                    diag.help("rewrite it as `if cond { loop { } }`");
                }
            },
        );
    }
}

struct HasBreakOrReturnVisitor {
    has_break_or_return: bool,
}

impl<'tcx> Visitor<'tcx> for HasBreakOrReturnVisitor {
    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if self.has_break_or_return {
            return;
        }

        match expr.kind {
            ExprKind::Ret(_) | ExprKind::Break(_, _) => {
                self.has_break_or_return = true;
                return;
            },
            _ => {},
        }

        walk_expr(self, expr);
    }
}

/// Collects the set of variables in an expression
/// Stops analysis if a function call is found
/// Note: In some cases such as `self`, there are no mutable annotation,
/// All variables definition IDs are collected
struct VarCollectorVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    ids: HirIdSet,
    def_ids: DefIdMap<bool>,
    skip: bool,
}

impl<'a, 'tcx> VarCollectorVisitor<'a, 'tcx> {
    fn insert_def_id(&mut self, ex: &'tcx Expr<'_>) {
        if_chain! {
            if let ExprKind::Path(ref qpath) = ex.kind;
            if let QPath::Resolved(None, _) = *qpath;
            then {
                match self.cx.qpath_res(qpath, ex.hir_id) {
                    Res::Local(hir_id) => {
                        self.ids.insert(hir_id);
                    },
                    Res::Def(DefKind::Static(_), def_id) => {
                        let mutable = self.cx.tcx.is_mutable_static(def_id);
                        self.def_ids.insert(def_id, mutable);
                    },
                    _ => {},
                }
            }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for VarCollectorVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, ex: &'tcx Expr<'_>) {
        match ex.kind {
            ExprKind::Path(_) => self.insert_def_id(ex),
            // If there is any function/method call… we just stop analysis
            ExprKind::Call(..) | ExprKind::MethodCall(..) => self.skip = true,

            _ => walk_expr(self, ex),
        }
    }
}
