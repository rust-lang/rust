use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::macros::{is_panic, matching_root_macro_call, root_macro_call};
use clippy_utils::source::{indent_of, reindent_multiline, snippet};
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{SpanlessEq, higher, is_trait_method, path_to_local_id, peel_blocks, sym};
use hir::{Body, HirId, MatchSource, Pat};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::{Closure, Expr, ExprKind, PatKind, PathSegment, QPath, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty::adjustment::Adjust;
use rustc_span::Span;
use rustc_span::symbol::{Ident, Symbol};

use super::{MANUAL_FILTER_MAP, MANUAL_FIND_MAP, OPTION_FILTER_MAP, RESULT_FILTER_MAP};

fn is_method(cx: &LateContext<'_>, expr: &Expr<'_>, method_name: Symbol) -> bool {
    match &expr.kind {
        ExprKind::Path(QPath::TypeRelative(_, mname)) => mname.ident.name == method_name,
        ExprKind::Path(QPath::Resolved(_, segments)) => segments.segments.last().unwrap().ident.name == method_name,
        ExprKind::MethodCall(segment, _, _, _) => segment.ident.name == method_name,
        ExprKind::Closure(Closure { body, .. }) => {
            let body = cx.tcx.hir_body(*body);
            let closure_expr = peel_blocks(body.value);
            match closure_expr.kind {
                ExprKind::MethodCall(PathSegment { ident, .. }, receiver, ..) => {
                    if ident.name == method_name
                        && let ExprKind::Path(path) = &receiver.kind
                        && let Res::Local(ref local) = cx.qpath_res(path, receiver.hir_id)
                        && !body.params.is_empty()
                    {
                        let arg_id = body.params[0].pat.hir_id;
                        return arg_id == *local;
                    }
                    false
                },
                _ => false,
            }
        },
        _ => false,
    }
}

fn is_option_filter_map(cx: &LateContext<'_>, filter_arg: &Expr<'_>, map_arg: &Expr<'_>) -> bool {
    is_method(cx, map_arg, sym::unwrap) && is_method(cx, filter_arg, sym::is_some)
}
fn is_ok_filter_map(cx: &LateContext<'_>, filter_arg: &Expr<'_>, map_arg: &Expr<'_>) -> bool {
    is_method(cx, map_arg, sym::unwrap) && is_method(cx, filter_arg, sym::is_ok)
}

#[derive(Debug, Copy, Clone)]
enum OffendingFilterExpr<'tcx> {
    /// `.filter(|opt| opt.is_some())`
    IsSome {
        /// The receiver expression
        receiver: &'tcx Expr<'tcx>,
        /// If `Some`, then this contains the span of an expression that possibly contains side
        /// effects: `.filter(|opt| side_effect(opt).is_some())`
        ///                         ^^^^^^^^^^^^^^^^
        ///
        /// We will use this later for warning the user that the suggested fix may change
        /// the behavior.
        side_effect_expr_span: Option<Span>,
    },
    /// `.filter(|res| res.is_ok())`
    IsOk {
        /// The receiver expression
        receiver: &'tcx Expr<'tcx>,
        /// See `IsSome`
        side_effect_expr_span: Option<Span>,
    },
    /// `.filter(|enum| matches!(enum, Enum::A(_)))`
    Matches {
        /// The `DefId` of the variant being matched
        variant_def_id: hir::def_id::DefId,
    },
}

#[derive(Debug)]
enum CalledMethod {
    OptionIsSome,
    ResultIsOk,
}

/// The result of checking a `map` call, returned by `OffendingFilterExpr::check_map_call`
#[derive(Debug)]
enum CheckResult<'tcx> {
    Method {
        map_arg: &'tcx Expr<'tcx>,
        /// The method that was called inside of `filter`
        method: CalledMethod,
        /// See `OffendingFilterExpr::IsSome`
        side_effect_expr_span: Option<Span>,
    },
    PatternMatching {
        /// The span of the variant being matched
        /// if let Some(s) = enum
        ///        ^^^^^^^
        variant_span: Span,
        /// if let Some(s) = enum
        ///             ^
        variant_ident: Ident,
    },
}

impl<'tcx> OffendingFilterExpr<'tcx> {
    pub fn check_map_call(
        &self,
        cx: &LateContext<'tcx>,
        map_body: &'tcx Body<'tcx>,
        map_param_id: HirId,
        filter_param_id: HirId,
        is_filter_param_ref: bool,
    ) -> Option<CheckResult<'tcx>> {
        match *self {
            OffendingFilterExpr::IsSome {
                receiver,
                side_effect_expr_span,
            }
            | OffendingFilterExpr::IsOk {
                receiver,
                side_effect_expr_span,
            } => {
                // check if closure ends with expect() or unwrap()
                if let ExprKind::MethodCall(seg, map_arg, ..) = map_body.value.kind
                    && matches!(seg.ident.name, sym::expect | sym::unwrap | sym::unwrap_or)
                    // .map(|y| f(y).copied().unwrap())
                    //          ~~~~
                    && let map_arg_peeled = match map_arg.kind {
                        ExprKind::MethodCall(method, original_arg, [], _) if acceptable_methods(method) => {
                            original_arg
                        },
                        _ => map_arg,
                    }
                    // .map(|y| y[.acceptable_method()].unwrap())
                    && let simple_equal = (path_to_local_id(receiver, filter_param_id)
                        && path_to_local_id(map_arg_peeled, map_param_id))
                    && let eq_fallback = (|a: &Expr<'_>, b: &Expr<'_>| {
                        // in `filter(|x| ..)`, replace `*x` with `x`
                        let a_path = if !is_filter_param_ref
                            && let ExprKind::Unary(UnOp::Deref, expr_path) = a.kind
                        { expr_path } else { a };
                        // let the filter closure arg and the map closure arg be equal
                        path_to_local_id(a_path, filter_param_id)
                            && path_to_local_id(b, map_param_id)
                            && cx.typeck_results().expr_ty_adjusted(a) == cx.typeck_results().expr_ty_adjusted(b)
                    })
                    && (simple_equal
                        || SpanlessEq::new(cx).expr_fallback(eq_fallback).eq_expr(receiver, map_arg_peeled))
                {
                    Some(CheckResult::Method {
                        map_arg,
                        side_effect_expr_span,
                        method: match self {
                            OffendingFilterExpr::IsSome { .. } => CalledMethod::OptionIsSome,
                            OffendingFilterExpr::IsOk { .. } => CalledMethod::ResultIsOk,
                            OffendingFilterExpr::Matches { .. } => unreachable!("only IsSome and IsOk can get here"),
                        },
                    })
                } else {
                    None
                }
            },
            OffendingFilterExpr::Matches { variant_def_id } => {
                let expr_uses_local = |pat: &Pat<'_>, expr: &Expr<'_>| {
                    if let PatKind::TupleStruct(QPath::Resolved(_, path), [subpat], _) = pat.kind
                        && let PatKind::Binding(_, local_id, ident, _) = subpat.kind
                        && path_to_local_id(expr.peel_blocks(), local_id)
                        && let Some(local_variant_def_id) = path.res.opt_def_id()
                        && local_variant_def_id == variant_def_id
                    {
                        Some((ident, pat.span))
                    } else {
                        None
                    }
                };

                // look for:
                // `if let Variant   (v) =         enum { v } else { unreachable!() }`
                //         ^^^^^^^    ^            ^^^^            ^^^^^^^^^^^^^^^^^^
                //    variant_span  variant_ident  scrutinee       else_ (blocks peeled later)
                // OR
                // `match enum {   Variant       (v) => v,      _ => unreachable!() }`
                //        ^^^^     ^^^^^^^        ^                  ^^^^^^^^^^^^^^
                //     scrutinee  variant_span  variant_ident        else_
                let (scrutinee, else_, variant_ident, variant_span) =
                    match higher::IfLetOrMatch::parse(cx, map_body.value) {
                        // For `if let` we want to check that the variant matching arm references the local created by
                        // its pattern
                        Some(higher::IfLetOrMatch::IfLet(sc, pat, then, Some(else_), ..))
                            if let Some((ident, span)) = expr_uses_local(pat, then) =>
                        {
                            (sc, else_, ident, span)
                        },
                        // For `match` we want to check that the "else" arm is the wildcard (`_`) pattern
                        // and that the variant matching arm references the local created by its pattern
                        Some(higher::IfLetOrMatch::Match(sc, [arm, wild_arm], MatchSource::Normal))
                            if let PatKind::Wild = wild_arm.pat.kind
                                && let Some((ident, span)) = expr_uses_local(arm.pat, arm.body.peel_blocks()) =>
                        {
                            (sc, wild_arm.body, ident, span)
                        },
                        _ => return None,
                    };

                if path_to_local_id(scrutinee, map_param_id)
                    // else branch should be a `panic!` or `unreachable!` macro call
                    && let Some(mac) = root_macro_call(else_.peel_blocks().span)
                    && (is_panic(cx, mac.def_id) || cx.tcx.opt_item_name(mac.def_id) == Some(sym::unreachable))
                {
                    Some(CheckResult::PatternMatching {
                        variant_span,
                        variant_ident,
                    })
                } else {
                    None
                }
            },
        }
    }

    fn hir(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>, filter_param_id: HirId) -> Option<Self> {
        if let ExprKind::MethodCall(path, receiver, [], _) = expr.kind
            && let Some(recv_ty) = cx.typeck_results().expr_ty(receiver).peel_refs().ty_adt_def()
        {
            // we still want to lint if the expression possibly contains side effects,
            // *but* it can't be machine-applicable then, because that can change the behavior of the program:
            // .filter(|x| effect(x).is_some()).map(|x| effect(x).unwrap())
            // vs.
            // .filter_map(|x| effect(x))
            //
            // the latter only calls `effect` once
            let side_effect_expr_span = receiver.can_have_side_effects().then_some(receiver.span);

            match (cx.tcx.get_diagnostic_name(recv_ty.did()), path.ident.name) {
                (Some(sym::Option), sym::is_some) => Some(Self::IsSome {
                    receiver,
                    side_effect_expr_span,
                }),
                (Some(sym::Result), sym::is_ok) => Some(Self::IsOk {
                    receiver,
                    side_effect_expr_span,
                }),
                _ => None,
            }
        } else if matching_root_macro_call(cx, expr.span, sym::matches_macro).is_some()
            // we know for a fact that the wildcard pattern is the second arm
            && let ExprKind::Match(scrutinee, [arm, _], _) = expr.kind
            && path_to_local_id(scrutinee, filter_param_id)
            && let PatKind::TupleStruct(QPath::Resolved(_, path), ..) = arm.pat.kind
            && let Some(variant_def_id) = path.res.opt_def_id()
        {
            Some(OffendingFilterExpr::Matches { variant_def_id })
        } else {
            None
        }
    }
}

/// is `filter(|x| x.is_some()).map(|x| x.unwrap())`
fn is_filter_some_map_unwrap(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    filter_recv: &Expr<'_>,
    filter_arg: &Expr<'_>,
    map_arg: &Expr<'_>,
) -> bool {
    let iterator = is_trait_method(cx, expr, sym::Iterator);
    let option = is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(filter_recv), sym::Option);

    (iterator || option) && is_option_filter_map(cx, filter_arg, map_arg)
}

/// is `filter(|x| x.is_ok()).map(|x| x.unwrap())`
fn is_filter_ok_map_unwrap(cx: &LateContext<'_>, expr: &Expr<'_>, filter_arg: &Expr<'_>, map_arg: &Expr<'_>) -> bool {
    // result has no filter, so we only check for iterators
    let iterator = is_trait_method(cx, expr, sym::Iterator);
    iterator && is_ok_filter_map(cx, filter_arg, map_arg)
}

/// lint use of `filter().map()` or `find().map()` for `Iterators`
#[allow(clippy::too_many_arguments)]
pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    filter_recv: &Expr<'_>,
    filter_arg: &Expr<'_>,
    filter_span: Span,
    map_recv: &Expr<'_>,
    map_arg: &Expr<'_>,
    map_span: Span,
    is_find: bool,
) {
    if is_filter_some_map_unwrap(cx, expr, filter_recv, filter_arg, map_arg) {
        span_lint_and_sugg(
            cx,
            OPTION_FILTER_MAP,
            filter_span.with_hi(expr.span.hi()),
            "`filter` for `Some` followed by `unwrap`",
            "consider using `flatten` instead",
            reindent_multiline("flatten()", true, indent_of(cx, map_span)),
            Applicability::MachineApplicable,
        );

        return;
    }

    if is_filter_ok_map_unwrap(cx, expr, filter_arg, map_arg) {
        span_lint_and_sugg(
            cx,
            RESULT_FILTER_MAP,
            filter_span.with_hi(expr.span.hi()),
            "`filter` for `Ok` followed by `unwrap`",
            "consider using `flatten` instead",
            reindent_multiline("flatten()", true, indent_of(cx, map_span)),
            Applicability::MachineApplicable,
        );

        return;
    }

    if let Some((map_param_ident, check_result)) = is_find_or_filter(cx, map_recv, filter_arg, map_arg) {
        let span = filter_span.with_hi(expr.span.hi());
        let (filter_name, lint) = if is_find {
            ("find", MANUAL_FIND_MAP)
        } else {
            ("filter", MANUAL_FILTER_MAP)
        };
        let msg = format!("`{filter_name}(..).map(..)` can be simplified as `{filter_name}_map(..)`");

        let (sugg, note_and_span, applicability) = match check_result {
            CheckResult::Method {
                map_arg,
                method,
                side_effect_expr_span,
            } => {
                let (to_opt, deref) = match method {
                    CalledMethod::ResultIsOk => (".ok()", String::new()),
                    CalledMethod::OptionIsSome => {
                        let derefs = cx
                            .typeck_results()
                            .expr_adjustments(map_arg)
                            .iter()
                            .filter(|adj| matches!(adj.kind, Adjust::Deref(_)))
                            .count();

                        ("", "*".repeat(derefs))
                    },
                };

                let sugg = format!(
                    "{filter_name}_map(|{map_param_ident}| {deref}{}{to_opt})",
                    snippet(cx, map_arg.span, ".."),
                );
                let (note_and_span, applicability) = if let Some(span) = side_effect_expr_span {
                    let note = "the suggestion might change the behavior of the program when merging `filter` and `map`, \
                        because this expression potentially contains side effects and will only execute once";

                    (Some((note, span)), Applicability::MaybeIncorrect)
                } else {
                    (None, Applicability::MachineApplicable)
                };

                (sugg, note_and_span, applicability)
            },
            CheckResult::PatternMatching {
                variant_span,
                variant_ident,
            } => {
                let pat = snippet(cx, variant_span, "<pattern>");

                (
                    format!(
                        "{filter_name}_map(|{map_param_ident}| match {map_param_ident} {{ \
                    {pat} => Some({variant_ident}), \
                    _ => None \
                }})"
                    ),
                    None,
                    Applicability::MachineApplicable,
                )
            },
        };
        span_lint_and_then(cx, lint, span, msg, |diag| {
            diag.span_suggestion(span, "try", sugg, applicability);

            if let Some((note, span)) = note_and_span {
                diag.span_note(span, note);
            }
        });
    }
}

fn is_find_or_filter<'a>(
    cx: &LateContext<'a>,
    map_recv: &Expr<'_>,
    filter_arg: &Expr<'_>,
    map_arg: &Expr<'_>,
) -> Option<(Ident, CheckResult<'a>)> {
    if is_trait_method(cx, map_recv, sym::Iterator)
        // filter(|x| ...is_some())...
        && let ExprKind::Closure(&Closure { body: filter_body_id, .. }) = filter_arg.kind
        && let filter_body = cx.tcx.hir_body(filter_body_id)
        && let [filter_param] = filter_body.params
        // optional ref pattern: `filter(|&x| ..)`
        && let (filter_pat, is_filter_param_ref) = if let PatKind::Ref(ref_pat, _) = filter_param.pat.kind {
            (ref_pat, true)
        } else {
            (filter_param.pat, false)
        }

        && let PatKind::Binding(_, filter_param_id, _, None) = filter_pat.kind
        && let Some(offending_expr) = OffendingFilterExpr::hir(cx, filter_body.value, filter_param_id)

        && let ExprKind::Closure(&Closure { body: map_body_id, .. }) = map_arg.kind
        && let map_body = cx.tcx.hir_body(map_body_id)
        && let [map_param] = map_body.params
        && let PatKind::Binding(_, map_param_id, map_param_ident, None) = map_param.pat.kind

        && let Some(check_result) =
            offending_expr.check_map_call(cx, map_body, map_param_id, filter_param_id, is_filter_param_ref)
    {
        return Some((map_param_ident, check_result));
    }
    None
}

fn acceptable_methods(method: &PathSegment<'_>) -> bool {
    matches!(
        method.ident.name,
        sym::clone
            | sym::as_ref
            | sym::copied
            | sym::cloned
            | sym::as_deref
            | sym::as_mut
            | sym::as_deref_mut
            | sym::to_owned
    )
}
