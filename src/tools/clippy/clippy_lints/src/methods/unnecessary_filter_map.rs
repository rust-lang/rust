use super::utils::clone_or_copy_needed;
use clippy_utils::diagnostics::span_lint;
use clippy_utils::res::{MaybeDef, MaybeQPath, MaybeResPath, MaybeTypeckRes};
use clippy_utils::sym;
use clippy_utils::ty::{is_copy, option_arg_ty};
use clippy_utils::usage::mutated_variables;
use clippy_utils::visitors::{Descend, for_each_expr_without_closures};
use core::ops::ControlFlow;
use rustc_hir as hir;
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_lint::LateContext;
use rustc_span::Span;
use std::fmt::Display;

use super::{UNNECESSARY_FILTER_MAP, UNNECESSARY_FIND_MAP};

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    arg: &'tcx hir::Expr<'tcx>,
    call_span: Span,
    kind: Kind,
) {
    if !cx.ty_based_def(expr).opt_parent(cx).is_diag_item(cx, sym::Iterator) {
        return;
    }

    if let hir::ExprKind::Closure(&hir::Closure { body, .. }) = arg.kind {
        let body = cx.tcx.hir_body(body);
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
            if kind.is_filter_map()
                && let hir::ExprKind::Call(expr, [arg]) = body.value.kind
                && expr.res(cx).ctor_parent(cx).is_lang_item(cx, OptionSome)
                && let hir::ExprKind::Path(_) = arg.kind
            {
                span_lint(
                    cx,
                    UNNECESSARY_FILTER_MAP,
                    call_span,
                    String::from("this call to `.filter_map(..)` is unnecessary"),
                );
                return;
            }
            match kind {
                Kind::FilterMap => "map(..)",
                Kind::FindMap => "map(..).next()",
            }
        } else if !found_mapping && !mutates_arg && (!clone_or_copy_needed || is_copy(cx, in_ty)) {
            let ty = cx.typeck_results().expr_ty(body.value);
            if option_arg_ty(cx, ty).is_some_and(|t| t == in_ty) {
                match kind {
                    Kind::FilterMap => "filter(..)",
                    Kind::FindMap => "find(..)",
                }
            } else {
                return;
            }
        } else {
            return;
        };
        span_lint(
            cx,
            match kind {
                Kind::FilterMap => UNNECESSARY_FILTER_MAP,
                Kind::FindMap => UNNECESSARY_FIND_MAP,
            },
            call_span,
            format!("this `.{kind}(..)` can be written more simply using `.{sugg}`"),
        );
    }
}

#[derive(Clone, Copy)]
pub(super) enum Kind {
    FilterMap,
    FindMap,
}

impl Kind {
    fn is_filter_map(self) -> bool {
        matches!(self, Self::FilterMap)
    }
}

impl Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FilterMap => f.write_str("filter_map"),
            Self::FindMap => f.write_str("find_map"),
        }
    }
}

// returns (found_mapping, found_filtering)
fn check_expression<'tcx>(cx: &LateContext<'tcx>, arg_id: hir::HirId, expr: &'tcx hir::Expr<'_>) -> (bool, bool) {
    match expr.kind {
        hir::ExprKind::Path(ref path)
            if cx
                .qpath_res(path, expr.hir_id)
                .ctor_parent(cx)
                .is_lang_item(cx, OptionNone) =>
        {
            // None
            (false, true)
        },
        hir::ExprKind::Call(func, args) => {
            if func.res(cx).ctor_parent(cx).is_lang_item(cx, OptionSome) {
                if args[0].res_local_id() == Some(arg_id) {
                    // Some(arg_id)
                    return (false, false);
                }
                // Some(not arg_id)
                return (true, false);
            }
            (true, true)
        },
        hir::ExprKind::MethodCall(segment, recv, [arg], _) => {
            if segment.ident.name == sym::then_some
                && cx.typeck_results().expr_ty(recv).is_bool()
                && arg.res_local_id() == Some(arg_id)
            {
                // bool.then_some(arg_id)
                (false, true)
            } else {
                // bool.then_some(not arg_id)
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
        _ => (true, true),
    }
}
