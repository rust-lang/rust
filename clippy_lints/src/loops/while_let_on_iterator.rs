use super::utils::{LoopNestVisitor, Nesting};
use super::WHILE_LET_ON_ITERATOR;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::implements_trait;
use clippy_utils::usage::mutated_variables;
use clippy_utils::{
    get_enclosing_block, is_refutable, is_trait_method, last_path_segment, path_to_local, path_to_local_id,
};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_block, walk_expr, NestedVisitorMap, Visitor};
use rustc_hir::{Expr, ExprKind, HirId, MatchSource, Node, PatKind};
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;
use rustc_span::symbol::sym;

pub(super) fn check(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
    if let ExprKind::Match(ref match_expr, ref arms, MatchSource::WhileLetDesugar) = expr.kind {
        let pat = &arms[0].pat.kind;
        if let (
            &PatKind::TupleStruct(ref qpath, ref pat_args, _),
            &ExprKind::MethodCall(ref method_path, _, ref method_args, _),
        ) = (pat, &match_expr.kind)
        {
            let iter_expr = &method_args[0];

            // Don't lint when the iterator is recreated on every iteration
            if_chain! {
                if let ExprKind::MethodCall(..) | ExprKind::Call(..) = iter_expr.kind;
                if let Some(iter_def_id) = cx.tcx.get_diagnostic_item(sym::Iterator);
                if implements_trait(cx, cx.typeck_results().expr_ty(iter_expr), iter_def_id, &[]);
                then {
                    return;
                }
            }

            let lhs_constructor = last_path_segment(qpath);
            if method_path.ident.name == sym::next
                && is_trait_method(cx, match_expr, sym::Iterator)
                && lhs_constructor.ident.name == sym::Some
                && (pat_args.is_empty()
                    || !is_refutable(cx, &pat_args[0])
                        && !is_used_inside(cx, iter_expr, &arms[0].body)
                        && !is_iterator_used_after_while_let(cx, iter_expr)
                        && !is_nested(cx, expr, &method_args[0]))
            {
                let mut applicability = Applicability::MachineApplicable;
                let iterator = snippet_with_applicability(cx, method_args[0].span, "_", &mut applicability);
                let loop_var = if pat_args.is_empty() {
                    "_".to_string()
                } else {
                    snippet_with_applicability(cx, pat_args[0].span, "_", &mut applicability).into_owned()
                };
                span_lint_and_sugg(
                    cx,
                    WHILE_LET_ON_ITERATOR,
                    expr.span.with_hi(match_expr.span.hi()),
                    "this loop could be written as a `for` loop",
                    "try",
                    format!("for {} in {}", loop_var, iterator),
                    applicability,
                );
            }
        }
    }
}

fn is_used_inside<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, container: &'tcx Expr<'_>) -> bool {
    let def_id = match path_to_local(expr) {
        Some(id) => id,
        None => return false,
    };
    if let Some(used_mutably) = mutated_variables(container, cx) {
        if used_mutably.contains(&def_id) {
            return true;
        }
    }
    false
}

fn is_iterator_used_after_while_let<'tcx>(cx: &LateContext<'tcx>, iter_expr: &'tcx Expr<'_>) -> bool {
    let def_id = match path_to_local(iter_expr) {
        Some(id) => id,
        None => return false,
    };
    let mut visitor = VarUsedAfterLoopVisitor {
        def_id,
        iter_expr_id: iter_expr.hir_id,
        past_while_let: false,
        var_used_after_while_let: false,
    };
    if let Some(enclosing_block) = get_enclosing_block(cx, def_id) {
        walk_block(&mut visitor, enclosing_block);
    }
    visitor.var_used_after_while_let
}

fn is_nested(cx: &LateContext<'_>, match_expr: &Expr<'_>, iter_expr: &Expr<'_>) -> bool {
    if_chain! {
        if let Some(loop_block) = get_enclosing_block(cx, match_expr.hir_id);
        let parent_node = cx.tcx.hir().get_parent_node(loop_block.hir_id);
        if let Some(Node::Expr(loop_expr)) = cx.tcx.hir().find(parent_node);
        then {
            return is_loop_nested(cx, loop_expr, iter_expr)
        }
    }
    false
}

fn is_loop_nested(cx: &LateContext<'_>, loop_expr: &Expr<'_>, iter_expr: &Expr<'_>) -> bool {
    let mut id = loop_expr.hir_id;
    let iter_id = if let Some(id) = path_to_local(iter_expr) {
        id
    } else {
        return true;
    };
    loop {
        let parent = cx.tcx.hir().get_parent_node(id);
        if parent == id {
            return false;
        }
        match cx.tcx.hir().find(parent) {
            Some(Node::Expr(expr)) => {
                if let ExprKind::Loop(..) = expr.kind {
                    return true;
                };
            },
            Some(Node::Block(block)) => {
                let mut block_visitor = LoopNestVisitor {
                    hir_id: id,
                    iterator: iter_id,
                    nesting: Nesting::Unknown,
                };
                walk_block(&mut block_visitor, block);
                if block_visitor.nesting == Nesting::RuledOut {
                    return false;
                }
            },
            Some(Node::Stmt(_)) => (),
            _ => {
                return false;
            },
        }
        id = parent;
    }
}

struct VarUsedAfterLoopVisitor {
    def_id: HirId,
    iter_expr_id: HirId,
    past_while_let: bool,
    var_used_after_while_let: bool,
}

impl<'tcx> Visitor<'tcx> for VarUsedAfterLoopVisitor {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if self.past_while_let {
            if path_to_local_id(expr, self.def_id) {
                self.var_used_after_while_let = true;
            }
        } else if self.iter_expr_id == expr.hir_id {
            self.past_while_let = true;
        }
        walk_expr(self, expr);
    }
    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}
