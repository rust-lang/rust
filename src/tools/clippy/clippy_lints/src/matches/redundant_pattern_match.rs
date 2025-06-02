use super::REDUNDANT_PATTERN_MATCHING;
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::walk_span_to_context;
use clippy_utils::sugg::{Sugg, make_unop};
use clippy_utils::ty::{is_type_diagnostic_item, needs_ordered_drop};
use clippy_utils::visitors::{any_temporaries_need_ordered_drop, for_each_expr_without_closures};
use clippy_utils::{higher, is_expn_of, is_trait_method, sym};
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::LangItem::{self, OptionNone, OptionSome, PollPending, PollReady, ResultErr, ResultOk};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Arm, Expr, ExprKind, Node, Pat, PatExpr, PatExprKind, PatKind, QPath, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, GenericArgKind, Ty};
use rustc_span::{Span, Symbol};
use std::fmt::Write;
use std::ops::ControlFlow;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
    if let Some(higher::WhileLet {
        let_pat,
        let_expr,
        let_span,
        ..
    }) = higher::WhileLet::hir(expr)
    {
        find_method_sugg_for_if_let(cx, expr, let_pat, let_expr, "while", false);
        find_if_let_true(cx, let_pat, let_expr, let_span);
    }
}

pub(super) fn check_if_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    pat: &'tcx Pat<'_>,
    scrutinee: &'tcx Expr<'_>,
    has_else: bool,
    let_span: Span,
) {
    find_if_let_true(cx, pat, scrutinee, let_span);
    find_method_sugg_for_if_let(cx, expr, pat, scrutinee, "if", has_else);
}

/// Looks for:
/// * `matches!(expr, true)`
pub fn check_matches_true<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    arm: &'tcx Arm<'_>,
    scrutinee: &'tcx Expr<'_>,
) {
    find_match_true(
        cx,
        arm.pat,
        scrutinee,
        expr.span.source_callsite(),
        "using `matches!` to pattern match a bool",
    );
}

/// Looks for any of:
/// * `if let true = ...`
/// * `if let false = ...`
/// * `while let true = ...`
fn find_if_let_true<'tcx>(cx: &LateContext<'tcx>, pat: &'tcx Pat<'_>, scrutinee: &'tcx Expr<'_>, let_span: Span) {
    find_match_true(cx, pat, scrutinee, let_span, "using `if let` to pattern match a bool");
}

/// Common logic between `find_if_let_true` and `check_matches_true`
fn find_match_true<'tcx>(
    cx: &LateContext<'tcx>,
    pat: &'tcx Pat<'_>,
    scrutinee: &'tcx Expr<'_>,
    span: Span,
    message: &'static str,
) {
    if let PatKind::Expr(lit) = pat.kind
        && let PatExprKind::Lit { lit, negated: false } = lit.kind
        && let LitKind::Bool(pat_is_true) = lit.node
    {
        let mut applicability = Applicability::MachineApplicable;

        let mut sugg = Sugg::hir_with_context(
            cx,
            scrutinee,
            scrutinee.span.source_callsite().ctxt(),
            "..",
            &mut applicability,
        );

        if !pat_is_true {
            sugg = make_unop("!", sugg);
        }

        span_lint_and_sugg(
            cx,
            REDUNDANT_PATTERN_MATCHING,
            span,
            message,
            "consider using the condition directly",
            sugg.into_string(),
            applicability,
        );
    }
}

// Extract the generic arguments out of a type
fn try_get_generic_ty(ty: Ty<'_>, index: usize) -> Option<Ty<'_>> {
    if let ty::Adt(_, subs) = ty.kind()
        && let Some(sub) = subs.get(index)
        && let GenericArgKind::Type(sub_ty) = sub.kind()
    {
        Some(sub_ty)
    } else {
        None
    }
}

fn find_method_and_type<'tcx>(
    cx: &LateContext<'tcx>,
    check_pat: &Pat<'_>,
    op_ty: Ty<'tcx>,
) -> Option<(&'static str, Ty<'tcx>)> {
    match check_pat.kind {
        PatKind::TupleStruct(ref qpath, args, rest) => {
            let is_wildcard = matches!(args.first().map(|p| &p.kind), Some(PatKind::Wild));
            let is_rest = matches!((args, rest.as_opt_usize()), ([], Some(_)));

            if is_wildcard || is_rest {
                let res = cx.typeck_results().qpath_res(qpath, check_pat.hir_id);
                let id = res.opt_def_id().map(|ctor_id| cx.tcx.parent(ctor_id))?;
                let lang_items = cx.tcx.lang_items();
                if Some(id) == lang_items.result_ok_variant() {
                    Some(("is_ok()", try_get_generic_ty(op_ty, 0).unwrap_or(op_ty)))
                } else if Some(id) == lang_items.result_err_variant() {
                    Some(("is_err()", try_get_generic_ty(op_ty, 1).unwrap_or(op_ty)))
                } else if Some(id) == lang_items.option_some_variant() {
                    Some(("is_some()", op_ty))
                } else if Some(id) == lang_items.poll_ready_variant() {
                    Some(("is_ready()", op_ty))
                } else if is_pat_variant(cx, check_pat, qpath, Item::Diag(sym::IpAddr, sym::V4)) {
                    Some(("is_ipv4()", op_ty))
                } else if is_pat_variant(cx, check_pat, qpath, Item::Diag(sym::IpAddr, sym::V6)) {
                    Some(("is_ipv6()", op_ty))
                } else {
                    None
                }
            } else {
                None
            }
        },
        PatKind::Expr(PatExpr {
            kind: PatExprKind::Path(path),
            hir_id,
            ..
        }) => {
            if let Res::Def(DefKind::Ctor(..), ctor_id) = cx.qpath_res(path, *hir_id)
                && let Some(variant_id) = cx.tcx.opt_parent(ctor_id)
            {
                let method = if cx.tcx.lang_items().option_none_variant() == Some(variant_id) {
                    "is_none()"
                } else if cx.tcx.lang_items().poll_pending_variant() == Some(variant_id) {
                    "is_pending()"
                } else {
                    return None;
                };
                // `None` and `Pending` don't have an inner type.
                Some((method, cx.tcx.types.unit))
            } else {
                None
            }
        },
        _ => None,
    }
}

fn find_method_sugg_for_if_let<'tcx>(
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
    let Some((good_method, inner_ty)) = find_method_and_type(cx, check_pat, op_ty) else {
        return;
    };

    // If this is the last expression in a block or there is an else clause then the whole
    // type needs to be considered, not just the inner type of the branch being matched on.
    // Note the last expression in a block is dropped after all local bindings.
    let check_ty = if has_else
        || (keyword == "if" && matches!(cx.tcx.hir_parent_iter(expr.hir_id).next(), Some((_, Node::Block(..)))))
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
    if keyword == "while"
        && let ExprKind::MethodCall(method_path, _, [], _) = let_expr.kind
        && method_path.ident.name == sym::next
        && is_trait_method(cx, let_expr, sym::Iterator)
    {
        return;
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
        format!("redundant pattern matching, consider using `{good_method}`"),
        |diag| {
            // if/while let ... = ... { ... }
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^
            let expr_span = expr.span;
            let ctxt = expr.span.ctxt();

            // if/while let ... = ... { ... }
            //                    ^^^
            let Some(res_span) = walk_span_to_context(result_expr.span.source_callsite(), ctxt) else {
                return;
            };

            // if/while let ... = ... { ... }
            // ^^^^^^^^^^^^^^^^^^^^^^
            let span = expr_span.until(res_span.shrink_to_hi());

            let mut app = if needs_drop {
                Applicability::MaybeIncorrect
            } else {
                Applicability::MachineApplicable
            };

            let sugg = Sugg::hir_with_context(cx, result_expr, ctxt, "_", &mut app)
                .maybe_paren()
                .to_string();

            diag.span_suggestion(span, "try", format!("{keyword} {sugg}.{good_method}"), app);

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

        if let Some((good_method, maybe_guard)) = found_good_method(cx, arms, node_pair) {
            let span = is_expn_of(expr.span, sym::matches).unwrap_or(expr.span.to(op.span));
            let result_expr = match &op.kind {
                ExprKind::AddrOf(_, _, borrowed) => borrowed,
                _ => op,
            };
            let mut app = Applicability::MachineApplicable;
            let receiver_sugg = Sugg::hir_with_applicability(cx, result_expr, "_", &mut app).maybe_paren();
            let mut sugg = format!("{receiver_sugg}.{good_method}");

            if let Some(guard) = maybe_guard {
                // wow, the HIR for match guards in `PAT if let PAT = expr && expr => ...` is annoying!
                // `guard` here is `Guard::If` with the let expression somewhere deep in the tree of exprs,
                // counter to the intuition that it should be `Guard::IfLet`, so we need another check
                // to see that there aren't any let chains anywhere in the guard, as that would break
                // if we suggest `t.is_none() && (let X = y && z)` for:
                // `match t { None if let X = y && z => true, _ => false }`
                let has_nested_let_chain = for_each_expr_without_closures(guard, |expr| {
                    if matches!(expr.kind, ExprKind::Let(..)) {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                })
                .is_some();

                if has_nested_let_chain {
                    return;
                }

                let guard = Sugg::hir(cx, guard, "..");
                let _ = write!(sugg, " && {}", guard.maybe_paren());
            }

            span_lint_and_sugg(
                cx,
                REDUNDANT_PATTERN_MATCHING,
                span,
                format!("redundant pattern matching, consider using `{good_method}`"),
                "try",
                sugg,
                app,
            );
        }
    }
}

fn found_good_method<'tcx>(
    cx: &LateContext<'_>,
    arms: &'tcx [Arm<'tcx>],
    node: (&PatKind<'_>, &PatKind<'_>),
) -> Option<(&'static str, Option<&'tcx Expr<'tcx>>)> {
    match node {
        (PatKind::TupleStruct(path_left, patterns_left, _), PatKind::TupleStruct(path_right, patterns_right, _))
            if patterns_left.len() == 1 && patterns_right.len() == 1 =>
        {
            if let (PatKind::Wild, PatKind::Wild) = (&patterns_left[0].kind, &patterns_right[0].kind) {
                find_good_method_for_match(
                    cx,
                    arms,
                    path_left,
                    path_right,
                    Item::Lang(ResultOk),
                    Item::Lang(ResultErr),
                    "is_ok()",
                    "is_err()",
                )
                .or_else(|| {
                    find_good_method_for_match(
                        cx,
                        arms,
                        path_left,
                        path_right,
                        Item::Diag(sym::IpAddr, sym::V4),
                        Item::Diag(sym::IpAddr, sym::V6),
                        "is_ipv4()",
                        "is_ipv6()",
                    )
                })
            } else {
                None
            }
        },
        (
            PatKind::TupleStruct(path_left, patterns, _),
            PatKind::Expr(PatExpr {
                kind: PatExprKind::Path(path_right),
                ..
            }),
        )
        | (
            PatKind::Expr(PatExpr {
                kind: PatExprKind::Path(path_left),
                ..
            }),
            PatKind::TupleStruct(path_right, patterns, _),
        ) if patterns.len() == 1 => {
            if let PatKind::Wild = patterns[0].kind {
                find_good_method_for_match(
                    cx,
                    arms,
                    path_left,
                    path_right,
                    Item::Lang(OptionSome),
                    Item::Lang(OptionNone),
                    "is_some()",
                    "is_none()",
                )
                .or_else(|| {
                    find_good_method_for_match(
                        cx,
                        arms,
                        path_left,
                        path_right,
                        Item::Lang(PollReady),
                        Item::Lang(PollPending),
                        "is_ready()",
                        "is_pending()",
                    )
                })
            } else {
                None
            }
        },
        (PatKind::TupleStruct(path_left, patterns, _), PatKind::Wild) if patterns.len() == 1 => {
            if let PatKind::Wild = patterns[0].kind {
                get_good_method(cx, arms, path_left)
            } else {
                None
            }
        },
        (
            PatKind::Expr(PatExpr {
                kind: PatExprKind::Path(path_left),
                ..
            }),
            PatKind::Wild,
        ) => get_good_method(cx, arms, path_left),
        _ => None,
    }
}

fn get_ident(path: &QPath<'_>) -> Option<rustc_span::symbol::Ident> {
    match path {
        QPath::Resolved(_, path) => {
            let name = path.segments[0].ident;
            Some(name)
        },
        _ => None,
    }
}

fn get_good_method<'tcx>(
    cx: &LateContext<'_>,
    arms: &'tcx [Arm<'tcx>],
    path_left: &QPath<'_>,
) -> Option<(&'static str, Option<&'tcx Expr<'tcx>>)> {
    if let Some(name) = get_ident(path_left) {
        let (expected_item_left, should_be_left, should_be_right) = match name.as_str() {
            "Ok" => (Item::Lang(ResultOk), "is_ok()", "is_err()"),
            "Err" => (Item::Lang(ResultErr), "is_err()", "is_ok()"),
            "Some" => (Item::Lang(OptionSome), "is_some()", "is_none()"),
            "None" => (Item::Lang(OptionNone), "is_none()", "is_some()"),
            "Ready" => (Item::Lang(PollReady), "is_ready()", "is_pending()"),
            "Pending" => (Item::Lang(PollPending), "is_pending()", "is_ready()"),
            "V4" => (Item::Diag(sym::IpAddr, sym::V4), "is_ipv4()", "is_ipv6()"),
            "V6" => (Item::Diag(sym::IpAddr, sym::V6), "is_ipv6()", "is_ipv4()"),
            _ => return None,
        };
        return find_good_method_for_matches_macro(
            cx,
            arms,
            path_left,
            expected_item_left,
            should_be_left,
            should_be_right,
        );
    }
    None
}

#[derive(Clone, Copy)]
enum Item {
    Lang(LangItem),
    Diag(Symbol, Symbol),
}

fn is_pat_variant(cx: &LateContext<'_>, pat: &Pat<'_>, path: &QPath<'_>, expected_item: Item) -> bool {
    let Some(id) = cx.typeck_results().qpath_res(path, pat.hir_id).opt_def_id() else {
        return false;
    };

    match expected_item {
        Item::Lang(expected_lang_item) => cx
            .tcx
            .lang_items()
            .get(expected_lang_item)
            .is_some_and(|expected_id| cx.tcx.parent(id) == expected_id),
        Item::Diag(expected_ty, expected_variant) => {
            let ty = cx.typeck_results().pat_ty(pat);

            if is_type_diagnostic_item(cx, ty, expected_ty) {
                let variant = ty
                    .ty_adt_def()
                    .expect("struct pattern type is not an ADT")
                    .variant_of_res(cx.qpath_res(path, pat.hir_id));

                return variant.name == expected_variant;
            }

            false
        },
    }
}

#[expect(clippy::too_many_arguments)]
fn find_good_method_for_match<'a, 'tcx>(
    cx: &LateContext<'_>,
    arms: &'tcx [Arm<'tcx>],
    path_left: &QPath<'_>,
    path_right: &QPath<'_>,
    expected_item_left: Item,
    expected_item_right: Item,
    should_be_left: &'a str,
    should_be_right: &'a str,
) -> Option<(&'a str, Option<&'tcx Expr<'tcx>>)> {
    let first_pat = arms[0].pat;
    let second_pat = arms[1].pat;

    let body_node_pair = if (is_pat_variant(cx, first_pat, path_left, expected_item_left))
        && (is_pat_variant(cx, second_pat, path_right, expected_item_right))
    {
        (&arms[0].body.kind, &arms[1].body.kind)
    } else if (is_pat_variant(cx, first_pat, path_left, expected_item_right))
        && (is_pat_variant(cx, second_pat, path_right, expected_item_left))
    {
        (&arms[1].body.kind, &arms[0].body.kind)
    } else {
        return None;
    };

    match body_node_pair {
        (ExprKind::Lit(lit_left), ExprKind::Lit(lit_right)) => match (&lit_left.node, &lit_right.node) {
            (LitKind::Bool(true), LitKind::Bool(false)) => Some((should_be_left, arms[0].guard)),
            (LitKind::Bool(false), LitKind::Bool(true)) => Some((should_be_right, arms[1].guard)),
            _ => None,
        },
        _ => None,
    }
}

fn find_good_method_for_matches_macro<'a, 'tcx>(
    cx: &LateContext<'_>,
    arms: &'tcx [Arm<'tcx>],
    path_left: &QPath<'_>,
    expected_item_left: Item,
    should_be_left: &'a str,
    should_be_right: &'a str,
) -> Option<(&'a str, Option<&'tcx Expr<'tcx>>)> {
    let first_pat = arms[0].pat;

    let body_node_pair = if is_pat_variant(cx, first_pat, path_left, expected_item_left) {
        (&arms[0].body.kind, &arms[1].body.kind)
    } else {
        return None;
    };

    match body_node_pair {
        (ExprKind::Lit(lit_left), ExprKind::Lit(lit_right)) => match (&lit_left.node, &lit_right.node) {
            (LitKind::Bool(true), LitKind::Bool(false)) => Some((should_be_left, arms[0].guard)),
            (LitKind::Bool(false), LitKind::Bool(true)) => Some((should_be_right, arms[1].guard)),
            _ => None,
        },
        _ => None,
    }
}
