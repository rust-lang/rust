use super::REDUNDANT_PATTERN_MATCHING;
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::{snippet, walk_span_to_context};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{is_type_diagnostic_item, needs_ordered_drop};
use clippy_utils::visitors::any_temporaries_need_ordered_drop;
use clippy_utils::{higher, is_expn_of, is_trait_method};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::LangItem::{self, OptionNone, OptionSome, PollPending, PollReady, ResultErr, ResultOk};
use rustc_hir::{Arm, Expr, ExprKind, Node, Pat, PatKind, QPath, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, GenericArgKind, Ty};
use rustc_span::{sym, Symbol};

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
                let Some(id) = res.opt_def_id().map(|ctor_id| cx.tcx.parent(ctor_id)) else {
                    return None;
                };
                let lang_items = cx.tcx.lang_items();
                if Some(id) == lang_items.result_ok_variant() {
                    Some(("is_ok()", try_get_generic_ty(op_ty, 0).unwrap_or(op_ty)))
                } else if Some(id) == lang_items.result_err_variant() {
                    Some(("is_err()", try_get_generic_ty(op_ty, 1).unwrap_or(op_ty)))
                } else if Some(id) == lang_items.option_some_variant() {
                    Some(("is_some()", op_ty))
                } else if Some(id) == lang_items.poll_ready_variant() {
                    Some(("is_ready()", op_ty))
                } else if is_pat_variant(cx, check_pat, qpath, Item::Diag(sym::IpAddr, sym!(V4))) {
                    Some(("is_ipv4()", op_ty))
                } else if is_pat_variant(cx, check_pat, qpath, Item::Diag(sym::IpAddr, sym!(V6))) {
                    Some(("is_ipv6()", op_ty))
                } else {
                    None
                }
            } else {
                None
            }
        },
        PatKind::Path(ref path) => {
            if let Res::Def(DefKind::Ctor(..), ctor_id) = cx.qpath_res(path, check_pat.hir_id)
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
    let Some((good_method, inner_ty)) = find_method_and_type(cx, check_pat, op_ty) else {
        return;
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
        &format!("redundant pattern matching, consider using `{good_method}`"),
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
                .maybe_par()
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

        if let Some(good_method) = found_good_method(cx, arms, node_pair) {
            let span = is_expn_of(expr.span, "matches").unwrap_or(expr.span.to(op.span));
            let result_expr = match &op.kind {
                ExprKind::AddrOf(_, _, borrowed) => borrowed,
                _ => op,
            };
            span_lint_and_sugg(
                cx,
                REDUNDANT_PATTERN_MATCHING,
                span,
                &format!("redundant pattern matching, consider using `{good_method}`"),
                "try",
                format!("{}.{good_method}", snippet(cx, result_expr.span, "_")),
                Applicability::MachineApplicable,
            );
        }
    }
}

fn found_good_method<'a>(
    cx: &LateContext<'_>,
    arms: &[Arm<'_>],
    node: (&PatKind<'_>, &PatKind<'_>),
) -> Option<&'a str> {
    match node {
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
                        Item::Diag(sym::IpAddr, sym!(V4)),
                        Item::Diag(sym::IpAddr, sym!(V6)),
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
        (PatKind::TupleStruct(ref path_left, patterns, _), PatKind::Wild) if patterns.len() == 1 => {
            if let PatKind::Wild = patterns[0].kind {
                get_good_method(cx, arms, path_left)
            } else {
                None
            }
        },
        (PatKind::Path(ref path_left), PatKind::Wild) => get_good_method(cx, arms, path_left),
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

fn get_good_method<'a>(cx: &LateContext<'_>, arms: &[Arm<'_>], path_left: &QPath<'_>) -> Option<&'a str> {
    if let Some(name) = get_ident(path_left) {
        return match name.as_str() {
            "Ok" => {
                find_good_method_for_matches_macro(cx, arms, path_left, Item::Lang(ResultOk), "is_ok()", "is_err()")
            },
            "Err" => {
                find_good_method_for_matches_macro(cx, arms, path_left, Item::Lang(ResultErr), "is_err()", "is_ok()")
            },
            "Some" => find_good_method_for_matches_macro(
                cx,
                arms,
                path_left,
                Item::Lang(OptionSome),
                "is_some()",
                "is_none()",
            ),
            "None" => find_good_method_for_matches_macro(
                cx,
                arms,
                path_left,
                Item::Lang(OptionNone),
                "is_none()",
                "is_some()",
            ),
            _ => None,
        };
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
            .map_or(false, |expected_id| cx.tcx.parent(id) == expected_id),
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
fn find_good_method_for_match<'a>(
    cx: &LateContext<'_>,
    arms: &[Arm<'_>],
    path_left: &QPath<'_>,
    path_right: &QPath<'_>,
    expected_item_left: Item,
    expected_item_right: Item,
    should_be_left: &'a str,
    should_be_right: &'a str,
) -> Option<&'a str> {
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
            (LitKind::Bool(true), LitKind::Bool(false)) => Some(should_be_left),
            (LitKind::Bool(false), LitKind::Bool(true)) => Some(should_be_right),
            _ => None,
        },
        _ => None,
    }
}

fn find_good_method_for_matches_macro<'a>(
    cx: &LateContext<'_>,
    arms: &[Arm<'_>],
    path_left: &QPath<'_>,
    expected_item_left: Item,
    should_be_left: &'a str,
    should_be_right: &'a str,
) -> Option<&'a str> {
    let first_pat = arms[0].pat;

    let body_node_pair = if is_pat_variant(cx, first_pat, path_left, expected_item_left) {
        (&arms[0].body.kind, &arms[1].body.kind)
    } else {
        return None;
    };

    match body_node_pair {
        (ExprKind::Lit(lit_left), ExprKind::Lit(lit_right)) => match (&lit_left.node, &lit_right.node) {
            (LitKind::Bool(true), LitKind::Bool(false)) => Some(should_be_left),
            (LitKind::Bool(false), LitKind::Bool(true)) => Some(should_be_right),
            _ => None,
        },
        _ => None,
    }
}
