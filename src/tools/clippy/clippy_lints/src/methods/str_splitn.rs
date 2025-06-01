use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::{span_lint_and_sugg, span_lint_and_then};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_context;
use clippy_utils::usage::local_used_after_expr;
use clippy_utils::visitors::{Descend, for_each_expr};
use clippy_utils::{is_diag_item_method, path_to_local_id, paths, sym};
use core::ops::ControlFlow;
use rustc_errors::Applicability;
use rustc_hir::{
    BindingMode, Expr, ExprKind, HirId, LangItem, LetStmt, MatchSource, Node, Pat, PatKind, QPath, Stmt, StmtKind,
};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::{Span, Symbol, SyntaxContext};

use super::{MANUAL_SPLIT_ONCE, NEEDLESS_SPLITN};

pub(super) fn check(
    cx: &LateContext<'_>,
    method_name: Symbol,
    expr: &Expr<'_>,
    self_arg: &Expr<'_>,
    pat_arg: &Expr<'_>,
    count: u128,
    msrv: Msrv,
) {
    if count < 2 || !cx.typeck_results().expr_ty_adjusted(self_arg).peel_refs().is_str() {
        return;
    }

    let needless = |usage_kind| match usage_kind {
        IterUsageKind::Nth(n) => count > n + 1,
        IterUsageKind::NextTuple => count > 2,
    };
    let manual = count == 2 && msrv.meets(cx, msrvs::STR_SPLIT_ONCE);

    match parse_iter_usage(cx, expr.span.ctxt(), cx.tcx.hir_parent_iter(expr.hir_id)) {
        Some(usage) if needless(usage.kind) => lint_needless(cx, method_name, expr, self_arg, pat_arg),
        Some(usage) if manual => check_manual_split_once(cx, method_name, expr, self_arg, pat_arg, &usage),
        None if manual => {
            check_manual_split_once_indirect(cx, method_name, expr, self_arg, pat_arg);
        },
        _ => {},
    }
}

fn lint_needless(cx: &LateContext<'_>, method_name: Symbol, expr: &Expr<'_>, self_arg: &Expr<'_>, pat_arg: &Expr<'_>) {
    let mut app = Applicability::MachineApplicable;
    let r = if method_name == sym::splitn { "" } else { "r" };

    span_lint_and_sugg(
        cx,
        NEEDLESS_SPLITN,
        expr.span,
        format!("unnecessary use of `{r}splitn`"),
        "try",
        format!(
            "{}.{r}split({})",
            snippet_with_context(cx, self_arg.span, expr.span.ctxt(), "..", &mut app).0,
            snippet_with_context(cx, pat_arg.span, expr.span.ctxt(), "..", &mut app).0,
        ),
        app,
    );
}

fn check_manual_split_once(
    cx: &LateContext<'_>,
    method_name: Symbol,
    expr: &Expr<'_>,
    self_arg: &Expr<'_>,
    pat_arg: &Expr<'_>,
    usage: &IterUsage,
) {
    let ctxt = expr.span.ctxt();
    let (msg, reverse) = if method_name == sym::splitn {
        ("manual implementation of `split_once`", false)
    } else {
        ("manual implementation of `rsplit_once`", true)
    };

    let mut app = Applicability::MachineApplicable;
    let self_snip = snippet_with_context(cx, self_arg.span, ctxt, "..", &mut app).0;
    let pat_snip = snippet_with_context(cx, pat_arg.span, ctxt, "..", &mut app).0;

    let sugg = match usage.kind {
        IterUsageKind::NextTuple => {
            if reverse {
                format!("{self_snip}.rsplit_once({pat_snip}).map(|(x, y)| (y, x))")
            } else {
                format!("{self_snip}.split_once({pat_snip})")
            }
        },
        IterUsageKind::Nth(1) => {
            let (r, field) = if reverse { ("r", 0) } else { ("", 1) };

            match usage.unwrap_kind {
                Some(UnwrapKind::Unwrap) => {
                    format!("{self_snip}.{r}split_once({pat_snip}).unwrap().{field}")
                },
                Some(UnwrapKind::QuestionMark) => {
                    format!("{self_snip}.{r}split_once({pat_snip})?.{field}")
                },
                None => {
                    format!("{self_snip}.{r}split_once({pat_snip}).map(|x| x.{field})")
                },
            }
        },
        IterUsageKind::Nth(_) => return,
    };

    span_lint_and_sugg(cx, MANUAL_SPLIT_ONCE, usage.span, msg, "try", sugg, app);
}

/// checks for
///
/// ```no_run
/// let mut iter = "a.b.c".splitn(2, '.');
/// let a = iter.next();
/// let b = iter.next();
/// ```
fn check_manual_split_once_indirect(
    cx: &LateContext<'_>,
    method_name: Symbol,
    expr: &Expr<'_>,
    self_arg: &Expr<'_>,
    pat_arg: &Expr<'_>,
) -> Option<()> {
    let ctxt = expr.span.ctxt();
    let mut parents = cx.tcx.hir_parent_iter(expr.hir_id);
    if let (_, Node::LetStmt(local)) = parents.next()?
        && let PatKind::Binding(BindingMode::MUT, iter_binding_id, _, None) = local.pat.kind
        && let (iter_stmt_id, Node::Stmt(_)) = parents.next()?
        && let (_, Node::Block(enclosing_block)) = parents.next()?
        && let mut stmts = enclosing_block
            .stmts
            .iter()
            .skip_while(|stmt| stmt.hir_id != iter_stmt_id)
            .skip(1)
        && let first = indirect_usage(cx, stmts.next()?, iter_binding_id, ctxt)?
        && let second = indirect_usage(cx, stmts.next()?, iter_binding_id, ctxt)?
        && first.unwrap_kind == second.unwrap_kind
        && first.name != second.name
        && !local_used_after_expr(cx, iter_binding_id, second.init_expr)
    {
        let (r, lhs, rhs) = if method_name == sym::splitn {
            ("", first.name, second.name)
        } else {
            ("r", second.name, first.name)
        };
        let msg = format!("manual implementation of `{r}split_once`");

        let mut app = Applicability::MachineApplicable;
        let self_snip = snippet_with_context(cx, self_arg.span, ctxt, "..", &mut app).0;
        let pat_snip = snippet_with_context(cx, pat_arg.span, ctxt, "..", &mut app).0;

        span_lint_and_then(cx, MANUAL_SPLIT_ONCE, local.span, msg, |diag| {
            diag.span_label(first.span, "first usage here");
            diag.span_label(second.span, "second usage here");

            let unwrap = match first.unwrap_kind {
                UnwrapKind::Unwrap => ".unwrap()",
                UnwrapKind::QuestionMark => "?",
            };

            // Add a multipart suggestion
            diag.multipart_suggestion(
                format!("replace with `{r}split_once`"),
                vec![
                    (
                        local.span,
                        format!("let ({lhs}, {rhs}) = {self_snip}.{r}split_once({pat_snip}){unwrap};"),
                    ),
                    (first.span, String::new()),  // Remove the first usage
                    (second.span, String::new()), // Remove the second usage
                ],
                app,
            );
        });
    }

    Some(())
}

#[derive(Debug)]
struct IndirectUsage<'a> {
    name: Symbol,
    span: Span,
    init_expr: &'a Expr<'a>,
    unwrap_kind: UnwrapKind,
}

/// returns `Some(IndirectUsage)` for e.g.
///
/// ```ignore
/// let name = binding.next()?;
/// let name = binding.next().unwrap();
/// ```
fn indirect_usage<'tcx>(
    cx: &LateContext<'tcx>,
    stmt: &Stmt<'tcx>,
    binding: HirId,
    ctxt: SyntaxContext,
) -> Option<IndirectUsage<'tcx>> {
    if let StmtKind::Let(&LetStmt {
        pat: Pat {
            kind: PatKind::Binding(BindingMode::NONE, _, ident, None),
            ..
        },
        init: Some(init_expr),
        hir_id: local_hir_id,
        ..
    }) = stmt.kind
    {
        let mut path_to_binding = None;
        let _: Option<!> = for_each_expr(cx, init_expr, |e| {
            if path_to_local_id(e, binding) {
                path_to_binding = Some(e);
            }
            ControlFlow::Continue(Descend::from(path_to_binding.is_none()))
        });

        let mut parents = cx.tcx.hir_parent_iter(path_to_binding?.hir_id);
        let iter_usage = parse_iter_usage(cx, ctxt, &mut parents)?;

        let (parent_id, _) = parents.find(|(_, node)| {
            !matches!(
                node,
                Node::Expr(Expr {
                    kind: ExprKind::Match(.., MatchSource::TryDesugar(_)),
                    ..
                })
            )
        })?;

        if let IterUsage {
            kind: IterUsageKind::Nth(0),
            unwrap_kind: Some(unwrap_kind),
            ..
        } = iter_usage
            && parent_id == local_hir_id
        {
            return Some(IndirectUsage {
                name: ident.name,
                span: stmt.span,
                init_expr,
                unwrap_kind,
            });
        }
    }

    None
}

#[derive(Debug, Clone, Copy)]
enum IterUsageKind {
    Nth(u128),
    NextTuple,
}

#[derive(Debug, PartialEq, Eq)]
enum UnwrapKind {
    Unwrap,
    QuestionMark,
}

#[derive(Debug)]
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
) -> Option<IterUsage> {
    let (kind, span) = match iter.next() {
        Some((_, Node::Expr(e))) if e.span.ctxt() == ctxt => {
            let ExprKind::MethodCall(name, _, args, _) = e.kind else {
                return None;
            };
            let did = cx.typeck_results().type_dependent_def_id(e.hir_id)?;
            let iter_id = cx.tcx.get_diagnostic_item(sym::Iterator)?;

            match (name.ident.as_str(), args) {
                ("next", []) if cx.tcx.trait_of_item(did) == Some(iter_id) => (IterUsageKind::Nth(0), e.span),
                ("next_tuple", []) => {
                    return if paths::ITERTOOLS_NEXT_TUPLE.matches(cx, did)
                        && let ty::Adt(adt_def, subs) = cx.typeck_results().expr_ty(e).kind()
                        && cx.tcx.is_diagnostic_item(sym::Option, adt_def.did())
                        && let ty::Tuple(subs) = subs.type_at(0).kind()
                        && subs.len() == 2
                    {
                        Some(IterUsage {
                            kind: IterUsageKind::NextTuple,
                            unwrap_kind: None,
                            span: e.span,
                        })
                    } else {
                        None
                    };
                },
                ("nth" | "skip", [idx_expr]) if cx.tcx.trait_of_item(did) == Some(iter_id) => {
                    if let Some(Constant::Int(idx)) = ConstEvalCtxt::new(cx).eval(idx_expr) {
                        let span = if name.ident.as_str() == "nth" {
                            e.span
                        } else if let Some((_, Node::Expr(next_expr))) = iter.next()
                            && let ExprKind::MethodCall(next_name, _, [], _) = next_expr.kind
                            && next_name.ident.name == sym::next
                            && next_expr.span.ctxt() == ctxt
                            && let Some(next_id) = cx.typeck_results().type_dependent_def_id(next_expr.hir_id)
                            && cx.tcx.trait_of_item(next_id) == Some(iter_id)
                        {
                            next_expr.span
                        } else {
                            return None;
                        };
                        (IterUsageKind::Nth(idx), span)
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
                [_],
            ) => {
                let parent_span = e.span.parent_callsite().unwrap();
                if parent_span.ctxt() == ctxt {
                    (Some(UnwrapKind::QuestionMark), parent_span)
                } else {
                    (None, span)
                }
            },
            _ if e.span.ctxt() != ctxt => (None, span),
            ExprKind::MethodCall(name, _, [], _)
                if name.ident.name == sym::unwrap
                    && cx
                        .typeck_results()
                        .type_dependent_def_id(e.hir_id)
                        .is_some_and(|id| is_diag_item_method(cx, id, sym::Option)) =>
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
