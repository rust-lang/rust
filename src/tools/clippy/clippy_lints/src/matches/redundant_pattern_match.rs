use super::REDUNDANT_PATTERN_MATCHING;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::needs_ordered_drop;
use clippy_utils::visitors::any_temporaries_need_ordered_drop;
use clippy_utils::{higher, is_lang_ctor, is_trait_method, match_def_path, paths};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionNone, PollPending};
use rustc_hir::{Arm, Expr, ExprKind, Node, Pat, PatKind, QPath, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, subst::GenericArgKind, DefIdTree, Ty};
use rustc_span::sym;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
    if let Some(higher::WhileLet { let_pat, let_expr, .. }) = higher::WhileLet::hir(expr) {
        find_sugg_for_if_let(cx, expr, let_pat, let_expr, "while", false);
    }
}

pub(super) fn check_if_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    pat: &'tcx Pat<'_>,
    scrutinee: &'tcx Expr<'_>,
    has_else: bool,
) {
    find_sugg_for_if_let(cx, expr, pat, scrutinee, "if", has_else);
}

// Extract the generic arguments out of a type
fn try_get_generic_ty(ty: Ty<'_>, index: usize) -> Option<Ty<'_>> {
    if_chain! {
        if let ty::Adt(_, subs) = ty.kind();
        if let Some(sub) = subs.get(index);
        if let GenericArgKind::Type(sub_ty) = sub.unpack();
        then {
            Some(sub_ty)
        } else {
            None
        }
    }
}

fn find_sugg_for_if_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    let_pat: &Pat<'_>,
    let_expr: &'tcx Expr<'_>,
    keyword: &'static str,
    has_else: bool,
) {
    // also look inside refs
    // if we have &None for example, peel it so we can detect "if let None = x"
    let check_pat = match let_pat.kind {
        PatKind::Ref(inner, _mutability) => inner,
        _ => let_pat,
    };
    let op_ty = cx.typeck_results().expr_ty(let_expr);
    // Determine which function should be used, and the type contained by the corresponding
    // variant.
    let (good_method, inner_ty) = match check_pat.kind {
        PatKind::TupleStruct(ref qpath, [sub_pat], _) => {
            if let PatKind::Wild = sub_pat.kind {
                let res = cx.typeck_results().qpath_res(qpath, check_pat.hir_id);
                let Some(id) = res.opt_def_id().map(|ctor_id| cx.tcx.parent(ctor_id)) else { return };
                let lang_items = cx.tcx.lang_items();
                if Some(id) == lang_items.result_ok_variant() {
                    ("is_ok()", try_get_generic_ty(op_ty, 0).unwrap_or(op_ty))
                } else if Some(id) == lang_items.result_err_variant() {
                    ("is_err()", try_get_generic_ty(op_ty, 1).unwrap_or(op_ty))
                } else if Some(id) == lang_items.option_some_variant() {
                    ("is_some()", op_ty)
                } else if Some(id) == lang_items.poll_ready_variant() {
                    ("is_ready()", op_ty)
                } else if match_def_path(cx, id, &paths::IPADDR_V4) {
                    ("is_ipv4()", op_ty)
                } else if match_def_path(cx, id, &paths::IPADDR_V6) {
                    ("is_ipv6()", op_ty)
                } else {
                    return;
                }
            } else {
                return;
            }
        },
        PatKind::Path(ref path) => {
            let method = if is_lang_ctor(cx, path, OptionNone) {
                "is_none()"
            } else if is_lang_ctor(cx, path, PollPending) {
                "is_pending()"
            } else {
                return;
            };
            // `None` and `Pending` don't have an inner type.
            (method, cx.tcx.types.unit)
        },
        _ => return,
    };

    // If this is the last expression in a block or there is an else clause then the whole
    // type needs to be considered, not just the inner type of the branch being matched on.
    // Note the last expression in a block is dropped after all local bindings.
    let check_ty = if has_else
        || (keyword == "if" && matches!(cx.tcx.hir().parent_iter(expr.hir_id).next(), Some((_, Node::Block(..)))))
    {
        op_ty
    } else {
        inner_ty
    };

    // All temporaries created in the scrutinee expression are dropped at the same time as the
    // scrutinee would be, so they have to be considered as well.
    // e.g. in `if let Some(x) = foo.lock().unwrap().baz.as_ref() { .. }` the lock will be held
    // for the duration if body.
    let needs_drop = needs_ordered_drop(cx, check_ty) || any_temporaries_need_ordered_drop(cx, let_expr);

    // check that `while_let_on_iterator` lint does not trigger
    if_chain! {
        if keyword == "while";
        if let ExprKind::MethodCall(method_path, ..) = let_expr.kind;
        if method_path.ident.name == sym::next;
        if is_trait_method(cx, let_expr, sym::Iterator);
        then {
            return;
        }
    }

    let result_expr = match &let_expr.kind {
        ExprKind::AddrOf(_, _, borrowed) => borrowed,
        ExprKind::Unary(UnOp::Deref, deref) => deref,
        _ => let_expr,
    };

    span_lint_and_then(
        cx,
        REDUNDANT_PATTERN_MATCHING,
        let_pat.span,
        &format!("redundant pattern matching, consider using `{}`", good_method),
        |diag| {
            // if/while let ... = ... { ... }
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^
            let expr_span = expr.span;

            // if/while let ... = ... { ... }
            //                 ^^^
            let op_span = result_expr.span.source_callsite();

            // if/while let ... = ... { ... }
            // ^^^^^^^^^^^^^^^^^^^
            let span = expr_span.until(op_span.shrink_to_hi());

            let app = if needs_drop {
                Applicability::MaybeIncorrect
            } else {
                Applicability::MachineApplicable
            };

            let sugg = Sugg::hir_with_macro_callsite(cx, result_expr, "_")
                .maybe_par()
                .to_string();

            diag.span_suggestion(span, "try this", format!("{} {}.{}", keyword, sugg, good_method), app);

            if needs_drop {
                diag.note("this will change drop order of the result, as well as all temporaries");
                diag.note("add `#[allow(clippy::redundant_pattern_matching)]` if this is important");
            }
        },
    );
}

pub(super) fn check_match<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, op: &Expr<'_>, arms: &[Arm<'_>]) {
    if arms.len() == 2 {
        let node_pair = (&arms[0].pat.kind, &arms[1].pat.kind);

        let found_good_method = match node_pair {
            (
                PatKind::TupleStruct(ref path_left, patterns_left, _),
                PatKind::TupleStruct(ref path_right, patterns_right, _),
            ) if patterns_left.len() == 1 && patterns_right.len() == 1 => {
                if let (PatKind::Wild, PatKind::Wild) = (&patterns_left[0].kind, &patterns_right[0].kind) {
                    find_good_method_for_match(
                        cx,
                        arms,
                        path_left,
                        path_right,
                        &paths::RESULT_OK,
                        &paths::RESULT_ERR,
                        "is_ok()",
                        "is_err()",
                    )
                    .or_else(|| {
                        find_good_method_for_match(
                            cx,
                            arms,
                            path_left,
                            path_right,
                            &paths::IPADDR_V4,
                            &paths::IPADDR_V6,
                            "is_ipv4()",
                            "is_ipv6()",
                        )
                    })
                } else {
                    None
                }
            },
            (PatKind::TupleStruct(ref path_left, patterns, _), PatKind::Path(ref path_right))
            | (PatKind::Path(ref path_left), PatKind::TupleStruct(ref path_right, patterns, _))
                if patterns.len() == 1 =>
            {
                if let PatKind::Wild = patterns[0].kind {
                    find_good_method_for_match(
                        cx,
                        arms,
                        path_left,
                        path_right,
                        &paths::OPTION_SOME,
                        &paths::OPTION_NONE,
                        "is_some()",
                        "is_none()",
                    )
                    .or_else(|| {
                        find_good_method_for_match(
                            cx,
                            arms,
                            path_left,
                            path_right,
                            &paths::POLL_READY,
                            &paths::POLL_PENDING,
                            "is_ready()",
                            "is_pending()",
                        )
                    })
                } else {
                    None
                }
            },
            _ => None,
        };

        if let Some(good_method) = found_good_method {
            let span = expr.span.to(op.span);
            let result_expr = match &op.kind {
                ExprKind::AddrOf(_, _, borrowed) => borrowed,
                _ => op,
            };
            span_lint_and_then(
                cx,
                REDUNDANT_PATTERN_MATCHING,
                expr.span,
                &format!("redundant pattern matching, consider using `{}`", good_method),
                |diag| {
                    diag.span_suggestion(
                        span,
                        "try this",
                        format!("{}.{}", snippet(cx, result_expr.span, "_"), good_method),
                        Applicability::MaybeIncorrect, // snippet
                    );
                },
            );
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn find_good_method_for_match<'a>(
    cx: &LateContext<'_>,
    arms: &[Arm<'_>],
    path_left: &QPath<'_>,
    path_right: &QPath<'_>,
    expected_left: &[&str],
    expected_right: &[&str],
    should_be_left: &'a str,
    should_be_right: &'a str,
) -> Option<&'a str> {
    let left_id = cx
        .typeck_results()
        .qpath_res(path_left, arms[0].pat.hir_id)
        .opt_def_id()?;
    let right_id = cx
        .typeck_results()
        .qpath_res(path_right, arms[1].pat.hir_id)
        .opt_def_id()?;
    let body_node_pair = if match_def_path(cx, left_id, expected_left) && match_def_path(cx, right_id, expected_right) {
        (&arms[0].body.kind, &arms[1].body.kind)
    } else if match_def_path(cx, right_id, expected_left) && match_def_path(cx, right_id, expected_right) {
        (&arms[1].body.kind, &arms[0].body.kind)
    } else {
        return None;
    };

    match body_node_pair {
        (ExprKind::Lit(ref lit_left), ExprKind::Lit(ref lit_right)) => match (&lit_left.node, &lit_right.node) {
            (LitKind::Bool(true), LitKind::Bool(false)) => Some(should_be_left),
            (LitKind::Bool(false), LitKind::Bool(true)) => Some(should_be_right),
            _ => None,
        },
        _ => None,
    }
}
