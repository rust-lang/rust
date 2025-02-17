use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{is_trait_method, path_to_local_id, peel_blocks, strip_pat_refs};
use rustc_ast::ast;
use rustc_data_structures::packed::Pu128;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::PatKind;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::{Span, sym};

use super::UNNECESSARY_FOLD;

/// Do we need to suggest turbofish when suggesting a replacement method?
/// Changing `fold` to `sum` needs it sometimes when the return type can't be
/// inferred. This checks for some common cases where it can be safely omitted
fn needs_turbofish(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    let parent = cx.tcx.parent_hir_node(expr.hir_id);

    // some common cases where turbofish isn't needed:
    // - assigned to a local variable with a type annotation
    if let hir::Node::LetStmt(local) = parent
        && local.ty.is_some()
    {
        return false;
    }

    // - part of a function call argument, can be inferred from the function signature (provided that
    //   the parameter is not a generic type parameter)
    if let hir::Node::Expr(parent_expr) = parent
        && let hir::ExprKind::Call(recv, args) = parent_expr.kind
        && let hir::ExprKind::Path(ref qpath) = recv.kind
        && let Some(fn_def_id) = cx.qpath_res(qpath, recv.hir_id).opt_def_id()
        && let fn_sig = cx.tcx.fn_sig(fn_def_id).skip_binder().skip_binder()
        && let Some(arg_pos) = args.iter().position(|arg| arg.hir_id == expr.hir_id)
        && let Some(ty) = fn_sig.inputs().get(arg_pos)
        && !matches!(ty.kind(), ty::Param(_))
    {
        return false;
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
) {
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

        && path_to_local_id(left_expr, first_arg_id)
        && (replacement.has_args || path_to_local_id(right_expr, second_arg_id))
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
            format!("{method}{turbofish}()", method = replacement.method_name,)
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
    }
}

pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &hir::Expr<'_>,
    init: &hir::Expr<'_>,
    acc: &hir::Expr<'_>,
    fold_span: Span,
) {
    // Check that this is a call to Iterator::fold rather than just some function called fold
    if !is_trait_method(cx, expr, sym::Iterator) {
        return;
    }

    // Check if the first argument to .fold is a suitable literal
    if let hir::ExprKind::Lit(lit) = init.kind {
        match lit.node {
            ast::LitKind::Bool(false) => {
                check_fold_with_op(
                    cx,
                    expr,
                    acc,
                    fold_span,
                    hir::BinOpKind::Or,
                    Replacement {
                        method_name: "any",
                        has_args: true,
                        has_generic_return: false,
                    },
                );
            },
            ast::LitKind::Bool(true) => {
                check_fold_with_op(
                    cx,
                    expr,
                    acc,
                    fold_span,
                    hir::BinOpKind::And,
                    Replacement {
                        method_name: "all",
                        has_args: true,
                        has_generic_return: false,
                    },
                );
            },
            ast::LitKind::Int(Pu128(0), _) => {
                check_fold_with_op(
                    cx,
                    expr,
                    acc,
                    fold_span,
                    hir::BinOpKind::Add,
                    Replacement {
                        method_name: "sum",
                        has_args: false,
                        has_generic_return: needs_turbofish(cx, expr),
                    },
                );
            },
            ast::LitKind::Int(Pu128(1), _) => {
                check_fold_with_op(
                    cx,
                    expr,
                    acc,
                    fold_span,
                    hir::BinOpKind::Mul,
                    Replacement {
                        method_name: "product",
                        has_args: false,
                        has_generic_return: needs_turbofish(cx, expr),
                    },
                );
            },
            _ => (),
        }
    }
}
