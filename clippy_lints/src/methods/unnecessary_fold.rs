use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{is_trait_method, path_to_local_id, peel_blocks, strip_pat_refs};
use if_chain::if_chain;
use rustc_ast::ast;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::PatKind;
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::{source_map::Span, sym};

use super::UNNECESSARY_FOLD;

/// Do we need to suggest turbofish when suggesting a replacement method?
/// Changing `fold` to `sum` needs it sometimes when the return type can't be
/// inferred. This checks for some common cases where it can be safely omitted
fn needs_turbofish(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    let parent = cx.tcx.hir().get_parent(expr.hir_id);

    // some common cases where turbofish isn't needed:
    // - assigned to a local variable with a type annotation
    if let hir::Node::Local(local) = parent
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
    if_chain! {
        // Extract the body of the closure passed to fold
        if let hir::ExprKind::Closure(&hir::Closure { body, .. }) = acc.kind;
        let closure_body = cx.tcx.hir().body(body);
        let closure_expr = peel_blocks(closure_body.value);

        // Check if the closure body is of the form `acc <op> some_expr(x)`
        if let hir::ExprKind::Binary(ref bin_op, left_expr, right_expr) = closure_expr.kind;
        if bin_op.node == op;

        // Extract the names of the two arguments to the closure
        if let [param_a, param_b] = closure_body.params;
        if let PatKind::Binding(_, first_arg_id, ..) = strip_pat_refs(param_a.pat).kind;
        if let PatKind::Binding(_, second_arg_id, second_arg_ident, _) = strip_pat_refs(param_b.pat).kind;

        if path_to_local_id(left_expr, first_arg_id);
        if replacement.has_args || path_to_local_id(right_expr, second_arg_id);

        then {
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
                format!(
                    "{method}{turbofish}()",
                    method = replacement.method_name,
                )
            };

            span_lint_and_sugg(
                cx,
                UNNECESSARY_FOLD,
                fold_span.with_hi(expr.span.hi()),
                // TODO #2371 don't suggest e.g., .any(|x| f(x)) if we can suggest .any(f)
                "this `.fold` can be written more succinctly using another method",
                "try",
                sugg,
                applicability,
            );
        }
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
                        has_args: true,
                        has_generic_return: false,
                        method_name: "any",
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
                        has_args: true,
                        has_generic_return: false,
                        method_name: "all",
                    },
                );
            },
            ast::LitKind::Int(0, _) => check_fold_with_op(
                cx,
                expr,
                acc,
                fold_span,
                hir::BinOpKind::Add,
                Replacement {
                    has_args: false,
                    has_generic_return: needs_turbofish(cx, expr),
                    method_name: "sum",
                },
            ),
            ast::LitKind::Int(1, _) => {
                check_fold_with_op(
                    cx,
                    expr,
                    acc,
                    fold_span,
                    hir::BinOpKind::Mul,
                    Replacement {
                        has_args: false,
                        has_generic_return: needs_turbofish(cx, expr),
                        method_name: "product",
                    },
                );
            },
            _ => (),
        }
    }
}
