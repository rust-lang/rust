use super::utils::clone_or_copy_needed;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::is_copy;
use clippy_utils::usage::mutated_variables;
use clippy_utils::visitors::{Descend, for_each_expr_without_closures};
use clippy_utils::{MaybePath, is_res_lang_ctor, is_trait_method, path_res, path_to_local_id};
use core::ops::ControlFlow;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_lint::LateContext;
use rustc_middle::query::Key;
use rustc_middle::ty;
use rustc_span::sym;

use super::{UNNECESSARY_FILTER_MAP, UNNECESSARY_FIND_MAP};

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>, arg: &'tcx hir::Expr<'tcx>, name: &str) {
    if !is_trait_method(cx, expr, sym::Iterator) {
        return;
    }

    if let hir::ExprKind::Closure(&hir::Closure { body, .. }) = arg.kind {
        let body = cx.tcx.hir().body(body);
        let arg_id = body.params[0].pat.hir_id;
        let mutates_arg = mutated_variables(body.value, cx).is_none_or(|used_mutably| used_mutably.contains(&arg_id));
        let (clone_or_copy_needed, _) = clone_or_copy_needed(cx, body.params[0].pat, body.value);

        let (mut found_mapping, mut found_filtering) = check_expression(cx, arg_id, body.value);

        let _: Option<!> = for_each_expr_without_closures(body.value, |e| {
            if let hir::ExprKind::Ret(Some(e)) = &e.kind {
                let (found_mapping_res, found_filtering_res) = check_expression(cx, arg_id, e);
                found_mapping |= found_mapping_res;
                found_filtering |= found_filtering_res;
                ControlFlow::Continue(Descend::No)
            } else {
                ControlFlow::Continue(Descend::Yes)
            }
        });
        let in_ty = cx.typeck_results().node_type(body.params[0].hir_id);
        let sugg = if !found_filtering {
            // Check if the closure is .filter_map(|x| Some(x))
            if name == "filter_map"
                && let hir::ExprKind::Call(expr, args) = body.value.kind
                && is_res_lang_ctor(cx, path_res(cx, expr), OptionSome)
                && arg_id.ty_def_id() == args[0].hir_id().ty_def_id()
                && let hir::ExprKind::Path(_) = args[0].kind
            {
                span_lint_and_sugg(
                    cx,
                    UNNECESSARY_FILTER_MAP,
                    expr.span,
                    format!("{name} is unnecessary"),
                    "try removing the filter_map",
                    String::new(),
                    Applicability::MaybeIncorrect,
                );
            }
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
        span_lint_and_sugg(
            cx,
            if name == "filter_map" {
                UNNECESSARY_FILTER_MAP
            } else {
                UNNECESSARY_FIND_MAP
            },
            expr.span,
            format!("this `.{name}` can be written more simply"),
            "try instead",
            sugg.to_string(),
            Applicability::MaybeIncorrect,
        );
    }
}

// returns (found_mapping, found_filtering)
fn check_expression<'tcx>(cx: &LateContext<'tcx>, arg_id: hir::HirId, expr: &'tcx hir::Expr<'_>) -> (bool, bool) {
    match expr.kind {
        hir::ExprKind::Call(func, args) => {
            if is_res_lang_ctor(cx, path_res(cx, func), OptionSome) {
                if path_to_local_id(&args[0], arg_id) {
                    return (false, false);
                }
                return (true, false);
            }
            (true, true)
        },
        hir::ExprKind::MethodCall(segment, recv, [arg], _) => {
            if segment.ident.name.as_str() == "then_some"
                && cx.typeck_results().expr_ty(recv).is_bool()
                && path_to_local_id(arg, arg_id)
            {
                (false, true)
            } else {
                (true, true)
            }
        },
        hir::ExprKind::Block(block, _) => block
            .expr
            .as_ref()
            .map_or((false, false), |expr| check_expression(cx, arg_id, expr)),
        hir::ExprKind::Match(_, arms, _) => {
            let mut found_mapping = false;
            let mut found_filtering = false;
            for arm in arms {
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
        hir::ExprKind::Path(ref path) if is_res_lang_ctor(cx, cx.qpath_res(path, expr.hir_id), OptionNone) => {
            (false, true)
        },
        _ => (true, true),
    }
}
