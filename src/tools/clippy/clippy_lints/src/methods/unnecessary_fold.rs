use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::{MaybeDef as _, MaybeQPath as _, MaybeResPath as _, MaybeTypeckRes as _};
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::is_copy;
use clippy_utils::visitors::for_each_expr;
use clippy_utils::{DefinedTy, ExprUseNode, get_expr_use_site, peel_blocks, strip_pat_refs};
use rustc_ast::ast;
use rustc_data_structures::packed::Pu128;
use rustc_errors::{Applicability, Diag};
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
fn needs_turbofish<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) -> bool {
    let use_site = get_expr_use_site(cx.tcx, cx.typeck_results(), expr.span.ctxt(), expr);
    if use_site.same_ctxt
        && let use_node = use_site.use_node(cx)
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
                    .any(|generic| generic.as_type().is_some_and(Ty::is_opaque)) =>
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
    is_short_circuiting: bool,
}

impl Replacement {
    fn default_applicability(&self) -> Applicability {
        if self.is_short_circuiting {
            Applicability::MaybeIncorrect
        } else {
            Applicability::MachineApplicable
        }
    }

    fn maybe_add_note(&self, diag: &mut Diag<'_, ()>) {
        if self.is_short_circuiting {
            diag.note(format!(
                "the `{}` method is short circuiting and may change the program semantics if the iterator has side effects",
                self.method_name
            ));
        }
    }

    fn maybe_turbofish(&self, ty: Ty<'_>) -> String {
        if self.has_generic_return {
            format!("::<{ty}>")
        } else {
            String::new()
        }
    }
}

fn get_triggered_expr_span(
    left_expr: &hir::Expr<'_>,
    right_expr: &hir::Expr<'_>,
    first_arg_id: hir::HirId,
    second_arg_id: hir::HirId,
    replacement: Replacement,
) -> Option<Span> {
    if left_expr.res_local_id() == Some(first_arg_id)
        && (replacement.has_args || right_expr.res_local_id() == Some(second_arg_id))
    {
        right_expr.span.into()
    }
    // https://github.com/rust-lang/rust-clippy/issues/16581
    else if right_expr.res_local_id() == Some(first_arg_id)
        && (replacement.has_args || left_expr.res_local_id() == Some(second_arg_id))
    {
        left_expr.span.into()
    } else {
        None
    }
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

        && let Some(triggered_expr_span) = get_triggered_expr_span(
            left_expr,
            right_expr,
            first_arg_id,
            second_arg_id,
            replacement
        )
    {
        let span = fold_span.with_hi(expr.span.hi());
        span_lint_and_then(
            cx,
            UNNECESSARY_FOLD,
            span,
            "this `.fold` can be written more succinctly using another method",
            |diag| {
                let mut applicability = replacement.default_applicability();
                let turbofish =
                    replacement.maybe_turbofish(cx.typeck_results().expr_ty_adjusted(right_expr).peel_refs());
                let (r_snippet, _) =
                    snippet_with_context(cx, triggered_expr_span, expr.span.ctxt(), "EXPR", &mut applicability);
                let sugg = if replacement.has_args {
                    format!(
                        "{method}{turbofish}(|{second_arg_ident}| {r_snippet})",
                        method = replacement.method_name,
                    )
                } else {
                    format!("{method}{turbofish}()", method = replacement.method_name)
                };

                diag.span_suggestion(span, "try", sugg, applicability);
                replacement.maybe_add_note(diag);
            },
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
) -> bool {
    // Extract the name of the function passed to `fold`
    if let Res::Def(DefKind::AssocFn, fn_did) = acc.res_if_named(cx, method)
        // Check if the function belongs to the operator
        && cx.tcx.is_diagnostic_item(method, fn_did)
    {
        let span = fold_span.with_hi(expr.span.hi());
        span_lint_and_then(
            cx,
            UNNECESSARY_FOLD,
            span,
            "this `.fold` can be written more succinctly using another method",
            |diag| {
                diag.span_suggestion(
                    span,
                    "try",
                    format!(
                        "{method}{turbofish}()",
                        method = replacement.method_name,
                        turbofish = replacement.maybe_turbofish(cx.typeck_results().expr_ty(expr))
                    ),
                    replacement.default_applicability(),
                );
                replacement.maybe_add_note(diag);
            },
        );
        return true;
    }
    false
}

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    recv: &'tcx hir::Expr<'tcx>,
    init: &hir::Expr<'_>,
    acc: &hir::Expr<'_>,
    fold_span: Span,
) {
    // Check that this is a call to Iterator::fold rather than just some function called fold
    if !cx.ty_based_def(expr).opt_parent(cx).is_diag_item(cx, sym::Iterator) {
        return;
    }

    if check_standard_fold(cx, expr, init, acc, fold_span) {
        return;
    }
    check_option_fold(cx, expr, recv, init, acc, fold_span);
}

/// Checks for the `any`/`all`/`sum`/`product` replacements. Returns `true`
/// when a lint was emitted.
fn check_standard_fold<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    init: &hir::Expr<'_>,
    acc: &hir::Expr<'_>,
    fold_span: Span,
) -> bool {
    // Check if the first argument to .fold is a suitable literal
    if let hir::ExprKind::Lit(lit) = init.kind {
        match lit.node {
            ast::LitKind::Bool(false) => {
                let replacement = Replacement {
                    method_name: "any",
                    has_args: true,
                    has_generic_return: false,
                    is_short_circuiting: true,
                };
                check_fold_with_op(cx, expr, acc, fold_span, hir::BinOpKind::Or, replacement)
            },
            ast::LitKind::Bool(true) => {
                let replacement = Replacement {
                    method_name: "all",
                    has_args: true,
                    has_generic_return: false,
                    is_short_circuiting: true,
                };
                check_fold_with_op(cx, expr, acc, fold_span, hir::BinOpKind::And, replacement)
            },
            ast::LitKind::Int(Pu128(0), _) => {
                let replacement = Replacement {
                    method_name: "sum",
                    has_args: false,
                    has_generic_return: needs_turbofish(cx, expr),
                    is_short_circuiting: false,
                };
                check_fold_with_op(cx, expr, acc, fold_span, hir::BinOpKind::Add, replacement)
                    || check_fold_with_method(cx, expr, acc, fold_span, sym::add, replacement)
            },
            ast::LitKind::Int(Pu128(1), _) => {
                let replacement = Replacement {
                    method_name: "product",
                    has_args: false,
                    has_generic_return: needs_turbofish(cx, expr),
                    is_short_circuiting: false,
                };
                check_fold_with_op(cx, expr, acc, fold_span, hir::BinOpKind::Mul, replacement)
                    || check_fold_with_method(cx, expr, acc, fold_span, sym::mul, replacement)
            },
            _ => false,
        }
    } else {
        false
    }
}

/// Checks whether `expr` is a path to a closure parameter.
///
/// Such a binding cannot be used as the substituted `init`: when folds are
/// nested, the enclosing closure may be an accumulator this lint removes in
/// another suggestion, which would leave the copied name unresolved.
fn is_closure_param(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    if let Some(binding_id) = expr.res_local_id() {
        let mut in_param = false;
        for (_, node) in cx.tcx.hir_parent_iter(binding_id) {
            match node {
                hir::Node::Pat(_) | hir::Node::PatField(_) => {},
                hir::Node::Param(_) => in_param = true,
                hir::Node::Expr(parent) => return in_param && matches!(parent.kind, hir::ExprKind::Closure(_)),
                _ => return false,
            }
        }
    }
    false
}

/// Folding over an `Option`'s iterator visits zero/one items.
/// This is the same as `map_or` if the `init` is `Copy` or a literal.
fn check_option_fold<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
    recv: &'tcx hir::Expr<'tcx>,
    init: &hir::Expr<'_>,
    acc: &hir::Expr<'_>,
    fold_span: Span,
) {
    let ctxt = expr.span.ctxt();
    match init.kind {
        hir::ExprKind::Lit(_) => {},
        hir::ExprKind::Path(_) if is_copy(cx, cx.typeck_results().expr_ty(init)) && !is_closure_param(cx, init) => {},
        _ => return,
    }
    if let hir::ExprKind::MethodCall(iter_path, option_expr, [], _) = recv.kind
        && let adapter = match iter_path.ident.name {
            sym::iter => "as_ref().",
            sym::iter_mut => "as_mut().",
            sym::into_iter => "",
            _ => return,
        }
        && cx.typeck_results().expr_ty(option_expr).is_diag_item(cx, sym::Option)
        && let hir::ExprKind::Closure(&hir::Closure { body, .. }) = acc.kind
        && let closure_body = cx.tcx.hir_body(body)
        && let [param_acc, param_item] = closure_body.params
        // The suggestion rewrites source spans, so bail out when any part
        // comes from a different syntax context.
        && recv.span.ctxt() == ctxt
        && param_acc.pat.span.ctxt() == ctxt
        && param_item.pat.span.ctxt() == ctxt
    {
        // Collect the accumulator uses to substitute with `init`.
        // A `mut` accumulator is likely reassigned in the closure body, which
        // would turn the substitution into nonsense like `0 += x`.
        let acc_id = match strip_pat_refs(param_acc.pat).kind {
            PatKind::Binding(hir::BindingMode::NONE, id, ..) => Some(id),
            PatKind::Wild => None,
            _ => return,
        };
        let mut acc_uses: Vec<Span> = Vec::new();
        let mut acc_uses_ok = true;
        if let Some(acc_id) = acc_id {
            for_each_expr(cx.tcx, closure_body.value, |sub_expr| {
                if sub_expr.res_local_id() == Some(acc_id) {
                    if sub_expr.span.ctxt() == ctxt {
                        acc_uses.push(sub_expr.span);
                    } else {
                        acc_uses_ok = false;
                    }
                }
                ControlFlow::<()>::Continue(())
            });
        }
        if !acc_uses_ok {
            return;
        }

        let mut applicability = Applicability::MachineApplicable;
        let (init_snippet, _) = snippet_with_context(cx, init.span, ctxt, "..", &mut applicability);

        // `.iter().fold(` -> `.as_ref().map_or(`, drop the accumulator
        // parameter, and substitute `init` for each use of the accumulator.
        acc_uses.sort();
        let mut parts = vec![
            (
                recv.span.with_lo(option_expr.span.hi()).with_hi(init.span.lo()),
                format!(".{adapter}map_or("),
            ),
            (param_acc.pat.span.with_hi(param_item.pat.span.lo()), String::new()),
        ];
        parts.extend(acc_uses.into_iter().map(|span| (span, init_snippet.to_string())));

        span_lint_and_then(
            cx,
            UNNECESSARY_FOLD,
            fold_span.with_hi(expr.span.hi()),
            "this `.fold` can be written more succinctly using another method",
            |diag| {
                diag.multipart_suggestion("try", parts, applicability);
            },
        );
    }
}
