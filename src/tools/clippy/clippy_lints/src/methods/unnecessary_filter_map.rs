use super::utils::clone_or_copy_needed;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::is_copy;
use clippy_utils::usage::mutated_variables;
use clippy_utils::{is_lang_ctor, is_trait_method, path_to_local_id};
use rustc_hir as hir;
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::sym;

use super::UNNECESSARY_FILTER_MAP;
use super::UNNECESSARY_FIND_MAP;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, arg: &hir::Expr<'_>, name: &str) {
    if !is_trait_method(cx, expr, sym::Iterator) {
        return;
    }

    if let hir::ExprKind::Closure(&hir::Closure { body, .. }) = arg.kind {
        let body = cx.tcx.hir().body(body);
        let arg_id = body.params[0].pat.hir_id;
        let mutates_arg = mutated_variables(body.value, cx).map_or(true, |used_mutably| used_mutably.contains(&arg_id));
        let (clone_or_copy_needed, _) = clone_or_copy_needed(cx, body.params[0].pat, body.value);

        let (mut found_mapping, mut found_filtering) = check_expression(cx, arg_id, body.value);

        let mut return_visitor = ReturnVisitor::new(cx, arg_id);
        return_visitor.visit_expr(body.value);
        found_mapping |= return_visitor.found_mapping;
        found_filtering |= return_visitor.found_filtering;

        let in_ty = cx.typeck_results().node_type(body.params[0].hir_id);
        let sugg = if !found_filtering {
            if name == "filter_map" { "map" } else { "map(..).next()" }
        } else if !found_mapping && !mutates_arg && (!clone_or_copy_needed || is_copy(cx, in_ty)) {
            match cx.typeck_results().expr_ty(body.value).kind() {
                ty::Adt(adt, subst)
                    if cx.tcx.is_diagnostic_item(sym::Option, adt.did()) && in_ty == subst.type_at(0) =>
                {
                    if name == "filter_map" { "filter" } else { "find" }
                },
                _ => return,
            }
        } else {
            return;
        };
        span_lint(
            cx,
            if name == "filter_map" {
                UNNECESSARY_FILTER_MAP
            } else {
                UNNECESSARY_FIND_MAP
            },
            expr.span,
            &format!("this `.{}` can be written more simply using `.{}`", name, sugg),
        );
    }
}

// returns (found_mapping, found_filtering)
fn check_expression<'tcx>(cx: &LateContext<'tcx>, arg_id: hir::HirId, expr: &'tcx hir::Expr<'_>) -> (bool, bool) {
    match &expr.kind {
        hir::ExprKind::Call(func, args) => {
            if let hir::ExprKind::Path(ref path) = func.kind {
                if is_lang_ctor(cx, path, OptionSome) {
                    if path_to_local_id(&args[0], arg_id) {
                        return (false, false);
                    }
                    return (true, false);
                }
            }
            (true, true)
        },
        hir::ExprKind::Block(block, _) => block
            .expr
            .as_ref()
            .map_or((false, false), |expr| check_expression(cx, arg_id, expr)),
        hir::ExprKind::Match(_, arms, _) => {
            let mut found_mapping = false;
            let mut found_filtering = false;
            for arm in *arms {
                let (m, f) = check_expression(cx, arg_id, arm.body);
                found_mapping |= m;
                found_filtering |= f;
            }
            (found_mapping, found_filtering)
        },
        // There must be an else_arm or there will be a type error
        hir::ExprKind::If(_, if_arm, Some(else_arm)) => {
            let if_check = check_expression(cx, arg_id, if_arm);
            let else_check = check_expression(cx, arg_id, else_arm);
            (if_check.0 | else_check.0, if_check.1 | else_check.1)
        },
        hir::ExprKind::Path(path) if is_lang_ctor(cx, path, OptionNone) => (false, true),
        _ => (true, true),
    }
}

struct ReturnVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    arg_id: hir::HirId,
    // Found a non-None return that isn't Some(input)
    found_mapping: bool,
    // Found a return that isn't Some
    found_filtering: bool,
}

impl<'a, 'tcx> ReturnVisitor<'a, 'tcx> {
    fn new(cx: &'a LateContext<'tcx>, arg_id: hir::HirId) -> ReturnVisitor<'a, 'tcx> {
        ReturnVisitor {
            cx,
            arg_id,
            found_mapping: false,
            found_filtering: false,
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for ReturnVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'_>) {
        if let hir::ExprKind::Ret(Some(expr)) = &expr.kind {
            let (found_mapping, found_filtering) = check_expression(self.cx, self.arg_id, expr);
            self.found_mapping |= found_mapping;
            self.found_filtering |= found_filtering;
        } else {
            walk_expr(self, expr);
        }
    }
}
