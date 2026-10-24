use std::borrow::Cow;

use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::eager_or_lazy::switch_to_eager_eval;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::res::{MaybeDef as _, MaybeResPath as _};
use clippy_utils::sugg::{Sugg, make_binop};
use clippy_utils::ty::{implements_trait, is_copy, needs_ordered_drop};
use clippy_utils::visitors::{any_temporaries_need_ordered_drop, is_local_used};
use clippy_utils::{get_parent_expr, is_from_proc_macro};
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, PatKind};
use rustc_lint::LateContext;
use rustc_span::{Span, sym};

use super::UNNECESSARY_MAP_OR;

pub(super) enum Variant {
    Ok,
    Some,
}
impl Variant {
    pub fn variant_name(&self) -> &'static str {
        match self {
            Variant::Ok => "Ok",
            Variant::Some => "Some",
        }
    }

    pub fn method_name(&self) -> &'static str {
        match self {
            Variant::Ok => "is_ok_and",
            Variant::Some => "is_some_and",
        }
    }
}

/// Evaluates `expr` and returns its value if it is a constant boolean.
fn bool_constant(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<bool> {
    let Some(Constant::Bool(value)) = ConstEvalCtxt::new(cx).eval(expr) else {
        return None;
    };
    Some(value)
}

/// Returns the constant boolean produced by a one-parameter closure.
fn closure_bool_constant(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<bool> {
    let ExprKind::Closure(closure) = expr.kind else {
        return None;
    };
    let body = cx.tcx.hir_body(closure.body);
    let [_] = body.params else {
        return None;
    };
    bool_constant(cx, body.value)
}

/// Checks whether a `Result::{map_or, map_or_else}` call is a variant query.
///
/// `expr` is the complete method call, `recv` is its `Result` receiver, `def` is the default
/// argument, and `map` is the mapping closure. `check_if_bool` accounts for the eager default in
/// `map_or` and the closure default in `map_or_else`.
fn check_result_variant_query<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    recv: &'tcx Expr<'tcx>,
    def: &'tcx Expr<'tcx>,
    map: &'tcx Expr<'tcx>,
    check_if_bool: impl FnOnce(&LateContext<'tcx>, &'tcx Expr<'tcx>) -> Option<bool>,
) -> bool {
    let ExprKind::MethodCall(path, _, _, call_span) = expr.kind else {
        return false;
    };
    let recv_ty = cx.typeck_results().expr_ty_adjusted(recv);
    if recv_ty.opt_diag_name(cx) != Some(sym::Result) {
        return false;
    }

    let def_bool = check_if_bool(cx, def);
    let Some((def_bool, map_bool)) = def_bool.zip(closure_bool_constant(cx, map)) else {
        return false;
    };
    if def_bool == map_bool || is_from_proc_macro(cx, expr) {
        return false;
    }

    let suggested_name = if map_bool { "is_ok" } else { "is_err" };
    let changes_drop_order = needs_ordered_drop(cx, recv_ty) || any_temporaries_need_ordered_drop(cx, recv);
    let applicability = if changes_drop_order {
        Applicability::MaybeIncorrect
    } else {
        Applicability::MachineApplicable
    };

    span_lint_and_then(
        cx,
        UNNECESSARY_MAP_OR,
        path.ident.span,
        format!("this `{}` can be simplified", path.ident.name),
        |diag| {
            diag.span_suggestion(
                call_span,
                format!("use `{suggested_name}` instead"),
                format!("{suggested_name}()"),
                applicability,
            );
            if changes_drop_order {
                diag.note("this will change drop order of the result, as well as all temporaries");
                diag.note("add `#[allow(clippy::unnecessary_map_or)]` if this is important");
            }
        },
    );
    true
}

/// Checks a `map_or` call for both `Result` variant queries and the existing `Option`/`Result`
/// simplifications.
///
/// `expr` is the complete method call, `recv` is its receiver, `def` is the eager default, and
/// `map` is the mapping closure. `method_span` identifies `map_or` in diagnostics, while `msrv`
/// controls which replacement methods can be suggested.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    recv: &'tcx Expr<'tcx>,
    def: &'tcx Expr<'tcx>,
    map: &'tcx Expr<'tcx>,
    method_span: Span,
    msrv: Msrv,
) {
    if check_result_variant_query(cx, expr, recv, def, map, bool_constant) {
        return;
    }

    let ExprKind::Lit(def_kind) = def.kind else {
        return;
    };
    let LitKind::Bool(def_bool) = def_kind.node else {
        return;
    };

    let typeck = cx.typeck_results();

    let recv_ty = typeck.expr_ty_adjusted(recv);

    let variant = match recv_ty.opt_diag_name(cx) {
        Some(sym::Option) => Variant::Some,
        Some(sym::Result) => Variant::Ok,
        Some(_) | None => return,
    };

    let ext_def_span = def.span.until(map.span);

    let (sugg, method, applicability): (_, Cow<'_, _>, _) = if typeck.expr_adjustments(recv).is_empty()
            && let ExprKind::Closure(map_closure) = map.kind
            && let closure_body = cx.tcx.hir_body(map_closure.body)
            && let closure_body_value = closure_body.value.peel_blocks()
            && let ExprKind::Binary(op, l, r) = closure_body_value.kind
            && let [param] = closure_body.params
            && let PatKind::Binding(_, hir_id, _, _) = param.pat.kind
            // checking that map_or is one of the following:
            // .map_or(false, |x| x == y)
            // .map_or(false, |x| y == x) - swapped comparison
            // .map_or(true, |x| x != y)
            // .map_or(true, |x| y != x) - swapped comparison
            && ((BinOpKind::Eq == op.node && !def_bool) || (BinOpKind::Ne == op.node && def_bool))
            && let non_binding_location = if l.res_local_id() == Some(hir_id) { r } else { l }
            && switch_to_eager_eval(cx, non_binding_location)
            // if it's both then that's a strange edge case and
            // we can just ignore it, since by default clippy will error on this
            && (l.res_local_id() == Some(hir_id)) != (r.res_local_id() == Some(hir_id))
            && !is_local_used(cx, non_binding_location, hir_id)
            && let l_ty = typeck.expr_ty(l)
            && l_ty == typeck.expr_ty(r)
            && let Some(partial_eq) = cx.tcx.lang_items().eq_trait()
            && implements_trait(cx, recv_ty, partial_eq, &[recv_ty.into()])
            && is_copy(cx, l_ty)
    {
        let wrap = variant.variant_name();

        // we may need to add parens around the suggestion
        // in case the parent expression has additional method calls,
        // since for example `Some(5).map_or(false, |x| x == 5).then(|| 1)`
        // being converted to `Some(5) == Some(5).then(|| 1)` isn't
        // the same thing

        let mut app = Applicability::MachineApplicable;
        let inner_non_binding = Sugg::NonParen(Cow::Owned(format!(
            "{wrap}({})",
            Sugg::hir_with_applicability(cx, non_binding_location, "", &mut app)
        )));

        let binop = make_binop(
            op.node,
            &Sugg::hir_with_applicability(cx, recv, "..", &mut app),
            &inner_non_binding,
        );

        let sugg = if let Some(parent_expr) = get_parent_expr(cx, expr) {
            if parent_expr.span.eq_ctxt(expr.span) {
                match parent_expr.kind {
                    ExprKind::Binary(..) | ExprKind::Unary(..) | ExprKind::Cast(..) => binop.maybe_paren(),
                    ExprKind::MethodCall(_, receiver, _, _) if receiver.hir_id == expr.hir_id => binop.maybe_paren(),
                    _ => binop,
                }
            } else {
                // if our parent expr is created by a macro, then it should be the one taking care of
                // parenthesising us if necessary
                binop
            }
        } else {
            binop
        }
        .into_string();

        (vec![(expr.span, sugg)], "a standard comparison".into(), app)
    } else if !def_bool && msrv.meets(cx, msrvs::OPTION_RESULT_IS_VARIANT_AND) {
        let suggested_name = variant.method_name();
        (
            vec![(method_span, suggested_name.into()), (ext_def_span, String::new())],
            format!("`{suggested_name}`").into(),
            Applicability::MachineApplicable,
        )
    } else if def_bool && matches!(variant, Variant::Some) && msrv.meets(cx, msrvs::IS_NONE_OR) {
        (
            vec![(method_span, "is_none_or".into()), (ext_def_span, String::new())],
            "`is_none_or`".into(),
            Applicability::MachineApplicable,
        )
    } else {
        return;
    };

    if is_from_proc_macro(cx, expr) {
        return;
    }

    span_lint_and_then(
        cx,
        UNNECESSARY_MAP_OR,
        expr.span,
        "this `map_or` can be simplified",
        |diag| {
            diag.multipart_suggestion(format!("use {method} instead"), sugg, applicability);
        },
    );
}

/// Checks a `map_or_else` call for a `Result` variant query.
///
/// `expr` is the complete method call, `recv` is its `Result` receiver, `def` is the lazy default
/// closure, and `map` is the mapping closure.
pub(super) fn check_map_or_else<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    recv: &'tcx Expr<'tcx>,
    def: &'tcx Expr<'tcx>,
    map: &'tcx Expr<'tcx>,
) {
    check_result_variant_query(cx, expr, recv, def, map, closure_bool_constant);
}
