use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use clippy_utils::{is_diag_item_method, match_def_path, paths};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, HirId, LangItem, Node, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, adjustment::Adjust};
use rustc_span::{symbol::sym, Span, SyntaxContext};

use super::MANUAL_SPLIT_ONCE;

pub(super) fn check_manual_split_once(
    cx: &LateContext<'_>,
    method_name: &str,
    expr: &Expr<'_>,
    self_arg: &Expr<'_>,
    pat_arg: &Expr<'_>,
) {
    if !cx.typeck_results().expr_ty_adjusted(self_arg).peel_refs().is_str() {
        return;
    }

    let ctxt = expr.span.ctxt();
    let (method_name, msg, reverse) = if method_name == "splitn" {
        ("split_once", "manual implementation of `split_once`", false)
    } else {
        ("rsplit_once", "manual implementation of `rsplit_once`", true)
    };
    let usage = match parse_iter_usage(cx, ctxt, cx.tcx.hir().parent_iter(expr.hir_id), reverse) {
        Some(x) => x,
        None => return,
    };

    let mut app = Applicability::MachineApplicable;
    let self_snip = snippet_with_context(cx, self_arg.span, ctxt, "..", &mut app).0;
    let pat_snip = snippet_with_context(cx, pat_arg.span, ctxt, "..", &mut app).0;

    let sugg = match usage.kind {
        IterUsageKind::NextTuple => {
            format!("{}.{}({})", self_snip, method_name, pat_snip)
        },
        IterUsageKind::RNextTuple => format!("{}.{}({}).map(|(x, y)| (y, x))", self_snip, method_name, pat_snip),
        IterUsageKind::Next | IterUsageKind::Second => {
            let self_deref = {
                let adjust = cx.typeck_results().expr_adjustments(self_arg);
                if adjust.len() < 2 {
                    String::new()
                } else if cx.typeck_results().expr_ty(self_arg).is_box()
                    || adjust
                        .iter()
                        .any(|a| matches!(a.kind, Adjust::Deref(Some(_))) || a.target.is_box())
                {
                    format!("&{}", "*".repeat(adjust.len().saturating_sub(1)))
                } else {
                    "*".repeat(adjust.len().saturating_sub(2))
                }
            };
            if matches!(usage.kind, IterUsageKind::Next) {
                match usage.unwrap_kind {
                    Some(UnwrapKind::Unwrap) => {
                        if reverse {
                            format!("{}.{}({}).unwrap().0", self_snip, method_name, pat_snip)
                        } else {
                            format!(
                                "{}.{}({}).map_or({}{}, |x| x.0)",
                                self_snip, method_name, pat_snip, self_deref, &self_snip
                            )
                        }
                    },
                    Some(UnwrapKind::QuestionMark) => {
                        format!(
                            "{}.{}({}).map_or({}{}, |x| x.0)",
                            self_snip, method_name, pat_snip, self_deref, &self_snip
                        )
                    },
                    None => {
                        format!(
                            "Some({}.{}({}).map_or({}{}, |x| x.0))",
                            &self_snip, method_name, pat_snip, self_deref, &self_snip
                        )
                    },
                }
            } else {
                match usage.unwrap_kind {
                    Some(UnwrapKind::Unwrap) => {
                        if reverse {
                            // In this case, no better suggestion is offered.
                            return;
                        }
                        format!("{}.{}({}).unwrap().1", self_snip, method_name, pat_snip)
                    },
                    Some(UnwrapKind::QuestionMark) => {
                        format!("{}.{}({})?.1", self_snip, method_name, pat_snip)
                    },
                    None => {
                        format!("{}.{}({}).map(|x| x.1)", self_snip, method_name, pat_snip)
                    },
                }
            }
        },
    };

    span_lint_and_sugg(cx, MANUAL_SPLIT_ONCE, usage.span, msg, "try this", sugg, app);
}

enum IterUsageKind {
    Next,
    Second,
    NextTuple,
    RNextTuple,
}

enum UnwrapKind {
    Unwrap,
    QuestionMark,
}

struct IterUsage {
    kind: IterUsageKind,
    unwrap_kind: Option<UnwrapKind>,
    span: Span,
}

#[allow(clippy::too_many_lines)]
fn parse_iter_usage<'tcx>(
    cx: &LateContext<'tcx>,
    ctxt: SyntaxContext,
    mut iter: impl Iterator<Item = (HirId, Node<'tcx>)>,
    reverse: bool,
) -> Option<IterUsage> {
    let (kind, span) = match iter.next() {
        Some((_, Node::Expr(e))) if e.span.ctxt() == ctxt => {
            let (name, args) = if let ExprKind::MethodCall(name, [_, args @ ..], _) = e.kind {
                (name, args)
            } else {
                return None;
            };
            let did = cx.typeck_results().type_dependent_def_id(e.hir_id)?;
            let iter_id = cx.tcx.get_diagnostic_item(sym::Iterator)?;

            match (name.ident.as_str(), args) {
                ("next", []) if cx.tcx.trait_of_item(did) == Some(iter_id) => {
                    if reverse {
                        (IterUsageKind::Second, e.span)
                    } else {
                        (IterUsageKind::Next, e.span)
                    }
                },
                ("next_tuple", []) => {
                    return if_chain! {
                        if match_def_path(cx, did, &paths::ITERTOOLS_NEXT_TUPLE);
                        if let ty::Adt(adt_def, subs) = cx.typeck_results().expr_ty(e).kind();
                        if cx.tcx.is_diagnostic_item(sym::Option, adt_def.did);
                        if let ty::Tuple(subs) = subs.type_at(0).kind();
                        if subs.len() == 2;
                        then {
                            Some(IterUsage {
                                kind: if reverse { IterUsageKind::RNextTuple } else { IterUsageKind::NextTuple },
                                span: e.span,
                                unwrap_kind: None
                            })
                        } else {
                            None
                        }
                    };
                },
                ("nth" | "skip", [idx_expr]) if cx.tcx.trait_of_item(did) == Some(iter_id) => {
                    if let Some((Constant::Int(idx), _)) = constant(cx, cx.typeck_results(), idx_expr) {
                        let span = if name.ident.as_str() == "nth" {
                            e.span
                        } else {
                            if_chain! {
                                if let Some((_, Node::Expr(next_expr))) = iter.next();
                                if let ExprKind::MethodCall(next_name, [_], _) = next_expr.kind;
                                if next_name.ident.name == sym::next;
                                if next_expr.span.ctxt() == ctxt;
                                if let Some(next_id) = cx.typeck_results().type_dependent_def_id(next_expr.hir_id);
                                if cx.tcx.trait_of_item(next_id) == Some(iter_id);
                                then {
                                    next_expr.span
                                } else {
                                    return None;
                                }
                            }
                        };
                        match if reverse { idx ^ 1 } else { idx } {
                            0 => (IterUsageKind::Next, span),
                            1 => (IterUsageKind::Second, span),
                            _ => return None,
                        }
                    } else {
                        return None;
                    }
                },
                _ => return None,
            }
        },
        _ => return None,
    };

    let (unwrap_kind, span) = if let Some((_, Node::Expr(e))) = iter.next() {
        match e.kind {
            ExprKind::Call(
                Expr {
                    kind: ExprKind::Path(QPath::LangItem(LangItem::TryTraitBranch, ..)),
                    ..
                },
                _,
            ) => {
                let parent_span = e.span.parent_callsite().unwrap();
                if parent_span.ctxt() == ctxt {
                    (Some(UnwrapKind::QuestionMark), parent_span)
                } else {
                    (None, span)
                }
            },
            _ if e.span.ctxt() != ctxt => (None, span),
            ExprKind::MethodCall(name, [_], _)
                if name.ident.name == sym::unwrap
                    && cx
                        .typeck_results()
                        .type_dependent_def_id(e.hir_id)
                        .map_or(false, |id| is_diag_item_method(cx, id, sym::Option)) =>
            {
                (Some(UnwrapKind::Unwrap), e.span)
            },
            _ => (None, span),
        }
    } else {
        (None, span)
    };

    Some(IterUsage {
        kind,
        unwrap_kind,
        span,
    })
}

use super::NEEDLESS_SPLITN;

pub(super) fn check_needless_splitn(
    cx: &LateContext<'_>,
    method_name: &str,
    expr: &Expr<'_>,
    self_arg: &Expr<'_>,
    pat_arg: &Expr<'_>,
    count: u128,
) {
    if !cx.typeck_results().expr_ty_adjusted(self_arg).peel_refs().is_str() {
        return;
    }
    let ctxt = expr.span.ctxt();
    let mut app = Applicability::MachineApplicable;
    let (reverse, message) = if method_name == "splitn" {
        (false, "unnecessary use of `splitn`")
    } else {
        (true, "unnecessary use of `rsplitn`")
    };
    if_chain! {
        if count >= 2;
        if check_iter(cx, ctxt, cx.tcx.hir().parent_iter(expr.hir_id), count);
        then {
            span_lint_and_sugg(
                cx,
                NEEDLESS_SPLITN,
                expr.span,
                message,
                "try this",
                format!(
                    "{}.{}({})",
                    snippet_with_context(cx, self_arg.span, ctxt, "..", &mut app).0,
                    if reverse {"rsplit"} else {"split"},
                    snippet_with_context(cx, pat_arg.span, ctxt, "..", &mut app).0
                ),
                app,
            );
        }
    }
}

fn check_iter<'tcx>(
    cx: &LateContext<'tcx>,
    ctxt: SyntaxContext,
    mut iter: impl Iterator<Item = (HirId, Node<'tcx>)>,
    count: u128,
) -> bool {
    match iter.next() {
        Some((_, Node::Expr(e))) if e.span.ctxt() == ctxt => {
            let (name, args) = if let ExprKind::MethodCall(name, [_, args @ ..], _) = e.kind {
                (name, args)
            } else {
                return false;
            };
            if_chain! {
                if let Some(did) = cx.typeck_results().type_dependent_def_id(e.hir_id);
                if let Some(iter_id) = cx.tcx.get_diagnostic_item(sym::Iterator);
                then {
                    match (name.ident.as_str(), args) {
                        ("next", []) if cx.tcx.trait_of_item(did) == Some(iter_id) => {
                            return true;
                        },
                        ("next_tuple", []) if count > 2 => {
                            return true;
                        },
                        ("nth", [idx_expr]) if cx.tcx.trait_of_item(did) == Some(iter_id) => {
                            if let Some((Constant::Int(idx), _)) = constant(cx, cx.typeck_results(), idx_expr) {
                                if count > idx + 1 {
                                    return true;
                                }
                            }
                        },
                        _ =>  return false,
                    }
                }
            }
        },
        _ => return false,
    };
    false
}
