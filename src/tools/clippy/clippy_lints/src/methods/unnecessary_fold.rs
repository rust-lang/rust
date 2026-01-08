use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::{MaybeDef, MaybeQPath, MaybeResPath, MaybeTypeckRes};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{DefinedTy, ExprUseNode, expr_use_ctxt, peel_blocks, strip_pat_refs};
use rustc_ast::ast;
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::PatKind;
use rustc_hir::def::{DefKind, Res};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::{Span, Symbol, sym};

use super::UNNECESSARY_FOLD;

/// Do we need to suggest turbofish when suggesting a replacement method?
/// Changing `fold` to `sum` needs it sometimes when the return type can't be
/// inferred. This checks for some common cases where it can be safely omitted
fn needs_turbofish<'tcx>(cx: &LateContext<'tcx>, expr: &hir::Expr<'tcx>) -> bool {
    let use_cx = expr_use_ctxt(cx, expr);
    if use_cx.same_ctxt
        && let use_node = use_cx.use_node(cx)
        && let Some(ty) = use_node.defined_ty(cx)
    {
        // some common cases where turbofish isn't needed:
        match (use_node, ty) {
            // - assigned to a local variable with a type annotation
            (ExprUseNode::LetStmt(_), _) => return false,

            // - part of a function call argument, can be inferred from the function signature (provided that the
            //   parameter is not a generic type parameter)
            (ExprUseNode::FnArg(..), DefinedTy::Mir { ty: arg_ty, .. })
                if !matches!(arg_ty.skip_binder().kind(), ty::Param(_)) =>
            {
                return false;
            },

            // - the final expression in the body of a function with a simple return type
            (ExprUseNode::Return(_), DefinedTy::Mir { ty: fn_return_ty, .. })
                if !fn_return_ty
                    .skip_binder()
                    .walk()
                    .any(|generic| generic.as_type().is_some_and(Ty::is_impl_trait)) =>
            {
                return false;
            },
            _ => {},
        }
    }

    // if it's neither of those, stay on the safe side and suggest turbofish,
    // even if it could work!
    true
}

#[derive(Copy, Clone)]
struct Replacement {
    method_name: &'static str,
    has_args: bool,
    has_generic_return: bool,
}

fn check_fold_with_op(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    acc: &hir::Expr<'_>,
    fold_span: Span,
    op: hir::BinOpKind,
    replacement: Replacement,
) -> bool {
    if let hir::ExprKind::Closure(&hir::Closure { body, .. }) = acc.kind
        // Extract the body of the closure passed to fold
        && let closure_body = cx.tcx.hir_body(body)
        && let closure_expr = peel_blocks(closure_body.value)

        // Check if the closure body is of the form `acc <op> some_expr(x)`
        && let hir::ExprKind::Binary(ref bin_op, left_expr, right_expr) = closure_expr.kind
        && bin_op.node == op

        // Extract the names of the two arguments to the closure
        && let [param_a, param_b] = closure_body.params
        && let PatKind::Binding(_, first_arg_id, ..) = strip_pat_refs(param_a.pat).kind
        && let PatKind::Binding(_, second_arg_id, second_arg_ident, _) = strip_pat_refs(param_b.pat).kind

        && left_expr.res_local_id() == Some(first_arg_id)
        && (replacement.has_args || right_expr.res_local_id() == Some(second_arg_id))
    {
        let mut applicability = Applicability::MachineApplicable;

        let turbofish = if replacement.has_generic_return {
            format!("::<{}>", cx.typeck_results().expr_ty_adjusted(right_expr).peel_refs())
        } else {
            String::new()
        };

        let sugg = if replacement.has_args {
            format!(
                "{method}{turbofish}(|{second_arg_ident}| {r})",
                method = replacement.method_name,
                r = snippet_with_applicability(cx, right_expr.span, "EXPR", &mut applicability),
            )
        } else {
            format!("{method}{turbofish}()", method = replacement.method_name)
        };

        span_lint_and_sugg(
            cx,
            UNNECESSARY_FOLD,
            fold_span.with_hi(expr.span.hi()),
            "this `.fold` can be written more succinctly using another method",
            "try",
            sugg,
            applicability,
        );
        return true;
    }
    false
}

fn check_fold_with_method(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    acc: &hir::Expr<'_>,
    fold_span: Span,
    method: Symbol,
    replacement: Replacement,
) {
    // Extract the name of the function passed to `fold`
    if let Res::Def(DefKind::AssocFn, fn_did) = acc.res_if_named(cx, method)
        // Check if the function belongs to the operator
        && cx.tcx.is_diagnostic_item(method, fn_did)
    {
        let applicability = Applicability::MachineApplicable;

        let turbofish = if replacement.has_generic_return {
            format!("::<{}>", cx.typeck_results().expr_ty(expr))
        } else {
            String::new()
        };

        span_lint_and_sugg(
            cx,
            UNNECESSARY_FOLD,
            fold_span.with_hi(expr.span.hi()),
            "this `.fold` can be written more succinctly using another method",
            "try",
            format!("{method}{turbofish}()", method = replacement.method_name),
            applicability,
        );
    }
}

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'tcx>,
    init: &hir::Expr<'_>,
    acc: &hir::Expr<'_>,
    fold_span: Span,
) {
    // Check that this is a call to Iterator::fold rather than just some function called fold
    if !cx.ty_based_def(expr).opt_parent(cx).is_diag_item(cx, sym::Iterator) {
        return;
    }

    // Check if the first argument to .fold is a suitable literal
    if let hir::ExprKind::Lit(lit) = init.kind {
        match lit.node {
            ast::LitKind::Bool(false) => {
                let replacement = Replacement {
                    method_name: "any",
                    has_args: true,
                    has_generic_return: false,
                };
                check_fold_with_op(cx, expr, acc, fold_span, hir::BinOpKind::Or, replacement);
            },
            ast::LitKind::Bool(true) => {
                let replacement = Replacement {
                    method_name: "all",
                    has_args: true,
                    has_generic_return: false,
                };
                check_fold_with_op(cx, expr, acc, fold_span, hir::BinOpKind::And, replacement);
            },
            ast::LitKind::Int(Pu128(0), _) => {
                let replacement = Replacement {
                    method_name: "sum",
                    has_args: false,
                    has_generic_return: needs_turbofish(cx, expr),
                };
                if !check_fold_with_op(cx, expr, acc, fold_span, hir::BinOpKind::Add, replacement) {
                    check_fold_with_method(cx, expr, acc, fold_span, sym::add, replacement);
                }
            },
            ast::LitKind::Int(Pu128(1), _) => {
                let replacement = Replacement {
                    method_name: "product",
                    has_args: false,
                    has_generic_return: needs_turbofish(cx, expr),
                };
                if !check_fold_with_op(cx, expr, acc, fold_span, hir::BinOpKind::Mul, replacement) {
                    check_fold_with_method(cx, expr, acc, fold_span, sym::mul, replacement);
                }
            },
            _ => (),
        }
    }
}
