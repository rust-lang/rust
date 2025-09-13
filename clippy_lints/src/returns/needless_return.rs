use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::source::snippet_with_context;
use clippy_utils::{
    binary_expr_needs_parentheses, is_from_proc_macro, leaks_droppable_temporary_with_limited_lifetime,
    span_contains_cfg, span_find_starting_semi, sym,
};
use rustc_ast::MetaItemInner;
use rustc_errors::Applicability;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{Body, Expr, ExprKind, HirId, LangItem, MatchSource, QPath, StmtKind};
use rustc_lint::{LateContext, Level, LintContext};
use rustc_middle::ty::{self, Ty};
use rustc_span::{BytePos, Pos, Span};
use std::borrow::Cow;
use std::fmt::Display;

use super::NEEDLESS_RETURN;

#[derive(PartialEq, Eq)]
enum RetReplacement<'tcx> {
    Empty,
    Block,
    Unit,
    NeedsPar(Cow<'tcx, str>, Applicability),
    Expr(Cow<'tcx, str>, Applicability),
}

impl RetReplacement<'_> {
    fn sugg_help(&self) -> &'static str {
        match self {
            Self::Empty | Self::Expr(..) => "remove `return`",
            Self::Block => "replace `return` with an empty block",
            Self::Unit => "replace `return` with a unit value",
            Self::NeedsPar(..) => "remove `return` and wrap the sequence with parentheses",
        }
    }

    fn applicability(&self) -> Applicability {
        match self {
            Self::Expr(_, ap) | Self::NeedsPar(_, ap) => *ap,
            _ => Applicability::MachineApplicable,
        }
    }
}

impl Display for RetReplacement<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str(""),
            Self::Block => f.write_str("{}"),
            Self::Unit => f.write_str("()"),
            Self::NeedsPar(inner, _) => write!(f, "({inner})"),
            Self::Expr(inner, _) => write!(f, "{inner}"),
        }
    }
}

pub(super) fn check_fn<'tcx>(cx: &LateContext<'tcx>, kind: FnKind<'tcx>, body: &'tcx Body<'tcx>, sp: Span) {
    if sp.from_expansion() {
        return;
    }

    match kind {
        FnKind::Closure => {
            // when returning without value in closure, replace this `return`
            // with an empty block to prevent invalid suggestion (see #6501)
            let replacement = if let ExprKind::Ret(None) = &body.value.kind {
                RetReplacement::Block
            } else {
                RetReplacement::Empty
            };
            check_final_expr(cx, body.value, vec![], replacement, None);
        },
        FnKind::ItemFn(..) | FnKind::Method(..) => {
            check_block_return(cx, &body.value.kind, sp, vec![]);
        },
    }
}

// if `expr` is a block, check if there are needless returns in it
fn check_block_return<'tcx>(cx: &LateContext<'tcx>, expr_kind: &ExprKind<'tcx>, sp: Span, mut semi_spans: Vec<Span>) {
    if let ExprKind::Block(block, _) = expr_kind {
        if let Some(block_expr) = block.expr {
            check_final_expr(cx, block_expr, semi_spans, RetReplacement::Empty, None);
        } else if let Some(stmt) = block.stmts.last() {
            if span_contains_cfg(
                cx,
                Span::between(
                    stmt.span,
                    cx.sess().source_map().end_point(block.span), // the closing brace of the block
                ),
            ) {
                return;
            }
            match stmt.kind {
                StmtKind::Expr(expr) => {
                    check_final_expr(cx, expr, semi_spans, RetReplacement::Empty, None);
                },
                StmtKind::Semi(semi_expr) => {
                    // Remove ending semicolons and any whitespace ' ' in between.
                    // Without `return`, the suggestion might not compile if the semicolon is retained
                    if let Some(semi_span) = stmt.span.trim_start(semi_expr.span) {
                        let semi_span_to_remove =
                            span_find_starting_semi(cx.sess().source_map(), semi_span.with_hi(sp.hi()));
                        semi_spans.push(semi_span_to_remove);
                    }
                    check_final_expr(cx, semi_expr, semi_spans, RetReplacement::Empty, None);
                },
                _ => (),
            }
        }
    }
}

fn check_final_expr<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    semi_spans: Vec<Span>, /* containing all the places where we would need to remove semicolons if finding an
                            * needless return */
    replacement: RetReplacement<'tcx>,
    match_ty_opt: Option<Ty<'_>>,
) {
    let peeled_drop_expr = expr.peel_drop_temps();
    match &peeled_drop_expr.kind {
        // simple return is always "bad"
        ExprKind::Ret(inner) => {
            // check if expr return nothing
            let ret_span = if inner.is_none() && replacement == RetReplacement::Empty {
                extend_span_to_previous_non_ws(cx, peeled_drop_expr.span)
            } else {
                peeled_drop_expr.span
            };

            let replacement = if let Some(inner_expr) = inner {
                // if desugar of `do yeet`, don't lint
                if let ExprKind::Call(path_expr, [_]) = inner_expr.kind
                    && let ExprKind::Path(QPath::LangItem(LangItem::TryTraitFromYeet, ..)) = path_expr.kind
                {
                    return;
                }

                let mut applicability = Applicability::MachineApplicable;
                let (snippet, _) = snippet_with_context(cx, inner_expr.span, ret_span.ctxt(), "..", &mut applicability);
                if binary_expr_needs_parentheses(inner_expr) {
                    RetReplacement::NeedsPar(snippet, applicability)
                } else {
                    RetReplacement::Expr(snippet, applicability)
                }
            } else {
                match match_ty_opt {
                    Some(match_ty) => {
                        match match_ty.kind() {
                            // If the code got till here with
                            // tuple not getting detected before it,
                            // then we are sure it's going to be Unit
                            // type
                            ty::Tuple(_) => RetReplacement::Unit,
                            // We don't want to anything in this case
                            // cause we can't predict what the user would
                            // want here
                            _ => return,
                        }
                    },
                    None => replacement,
                }
            };

            if inner.is_some_and(|inner| leaks_droppable_temporary_with_limited_lifetime(cx, inner)) {
                return;
            }

            if ret_span.from_expansion() || is_from_proc_macro(cx, expr) {
                return;
            }

            // Returns may be used to turn an expression into a statement in rustc's AST.
            // This allows the addition of attributes, like `#[allow]` (See: clippy#9361)
            // `#[expect(clippy::needless_return)]` needs to be handled separately to
            // actually fulfill the expectation (clippy::#12998)
            match cx.tcx.hir_attrs(expr.hir_id) {
                [] => {},
                [attr] => {
                    if matches!(Level::from_attr(attr), Some((Level::Expect, _)))
                        && let metas = attr.meta_item_list()
                        && let Some(lst) = metas
                        && let [MetaItemInner::MetaItem(meta_item), ..] = lst.as_slice()
                        && let [tool, lint_name] = meta_item.path.segments.as_slice()
                        && tool.ident.name == sym::clippy
                        && matches!(
                            lint_name.ident.name,
                            sym::needless_return | sym::style | sym::all | sym::warnings
                        )
                    {
                        // This is an expectation of the `needless_return` lint
                    } else {
                        return;
                    }
                },
                _ => return,
            }

            emit_return_lint(
                cx,
                peeled_drop_expr.span,
                ret_span,
                semi_spans,
                &replacement,
                expr.hir_id,
            );
        },
        ExprKind::If(_, then, else_clause_opt) => {
            check_block_return(cx, &then.kind, peeled_drop_expr.span, semi_spans.clone());
            if let Some(else_clause) = else_clause_opt {
                // The `RetReplacement` won't be used there as `else_clause` will be either a block or
                // a `if` expression.
                check_final_expr(cx, else_clause, semi_spans, RetReplacement::Empty, match_ty_opt);
            }
        },
        // a match expr, check all arms
        // an if/if let expr, check both exprs
        // note, if without else is going to be a type checking error anyways
        // (except for unit type functions) so we don't match it
        ExprKind::Match(_, arms, MatchSource::Normal) => {
            let match_ty = cx.typeck_results().expr_ty(peeled_drop_expr);
            for arm in *arms {
                check_final_expr(cx, arm.body, semi_spans.clone(), RetReplacement::Unit, Some(match_ty));
            }
        },
        // if it's a whole block, check it
        other_expr_kind => check_block_return(cx, other_expr_kind, peeled_drop_expr.span, semi_spans),
    }
}

fn emit_return_lint(
    cx: &LateContext<'_>,
    lint_span: Span,
    ret_span: Span,
    semi_spans: Vec<Span>,
    replacement: &RetReplacement<'_>,
    at: HirId,
) {
    span_lint_hir_and_then(
        cx,
        NEEDLESS_RETURN,
        at,
        lint_span,
        "unneeded `return` statement",
        |diag| {
            let suggestions = std::iter::once((ret_span, replacement.to_string()))
                .chain(semi_spans.into_iter().map(|span| (span, String::new())))
                .collect();

            diag.multipart_suggestion_verbose(replacement.sugg_help(), suggestions, replacement.applicability());
        },
    );
}

// Go backwards while encountering whitespace and extend the given Span to that point.
fn extend_span_to_previous_non_ws(cx: &LateContext<'_>, sp: Span) -> Span {
    if let Ok(prev_source) = cx.sess().source_map().span_to_prev_source(sp) {
        let ws = [b' ', b'\t', b'\n'];
        if let Some(non_ws_pos) = prev_source.bytes().rposition(|c| !ws.contains(&c)) {
            let len = prev_source.len() - non_ws_pos - 1;
            return sp.with_lo(sp.lo() - BytePos::from_usize(len));
        }
    }

    sp
}
