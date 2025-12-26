use std::iter;
use std::ops::ControlFlow;

use crate::internal_paths::MAYBE_DEF;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::res::{MaybeDef, MaybeTypeckRes};
use clippy_utils::source::{snippet_indent, snippet_with_applicability};
use clippy_utils::visitors::for_each_expr;
use clippy_utils::{eq_expr_value, if_sequence, sym};
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Block, Expr, ExprKind, Node, StmtKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_tool_lint! {
    /// ### What it does
    /// Checks for repeated use of `MaybeDef::is_diag_item`/`TyCtxt::is_diagnostic_item`;
    /// suggests to first call `MaybDef::opt_diag_name`/`TyCtxt::get_diagnostic_name` and then
    /// compare the output with all the `Symbol`s.
    ///
    /// ### Why is this bad?
    /// Each of such calls ultimately invokes the `diagnostic_items` query.
    /// While the query is cached, it's still better to avoid calling it multiple times if possible.
    ///
    /// ### Example
    /// ```no_run
    /// ty.is_diag_item(cx, sym::Option) || ty.is_diag_item(cx, sym::Result)
    /// cx.tcx.is_diagnostic_item(sym::Option, did) || cx.tcx.is_diagnostic_item(sym::Result, did)
    ///
    /// if ty.is_diag_item(cx, sym::Option) {
    ///     ..
    /// } else if ty.is_diag_item(cx, sym::Result) {
    ///     ..
    /// } else {
    ///     ..
    /// }
    ///
    /// if cx.tcx.is_diagnostic_item(sym::Option, did) {
    ///     ..
    /// } else if cx.tcx.is_diagnostic_item(sym::Result, did) {
    ///     ..
    /// } else {
    ///     ..
    /// }
    ///
    /// {
    ///     if ty.is_diag_item(cx, sym::Option) {
    ///         ..
    ///     }
    ///     if ty.is_diag_item(cx, sym::Result) {
    ///         ..
    ///     }
    /// }
    ///
    /// {
    ///     if cx.tcx.is_diagnostic_item(sym::Option, did) {
    ///         ..
    ///     }
    ///     if cx.tcx.is_diagnostic_item(sym::Result, did) {
    ///         ..
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// matches!(ty.opt_diag_name(cx), Some(sym::Option | sym::Result))
    /// matches!(cx.tcx.get_diagnostic_name(did), Some(sym::Option | sym::Result))
    ///
    /// match ty.opt_diag_name(cx) {
    ///     Some(sym::Option) => {
    ///         ..
    ///     }
    ///     Some(sym::Result) => {
    ///         ..
    ///     }
    ///     _ => {
    ///         ..
    ///     }
    /// }
    ///
    /// match cx.tcx.get_diagnostic_name(did) {
    ///     Some(sym::Option) => {
    ///         ..
    ///     }
    ///     Some(sym::Result) => {
    ///         ..
    ///     }
    ///     _ => {
    ///         ..
    ///     }
    /// }
    ///
    /// {
    ///     let name = ty.opt_diag_name(cx);
    ///     if name == Some(sym::Option) {
    ///         ..
    ///     }
    ///     if name == Some(sym::Result) {
    ///         ..
    ///     }
    /// }
    ///
    /// {
    ///     let name = cx.tcx.get_diagnostic_name(did);
    ///     if name == Some(sym::Option) {
    ///         ..
    ///     }
    ///     if name == Some(sym::Result) {
    ///         ..
    ///     }
    /// }
    /// ```
    pub clippy::REPEATED_IS_DIAGNOSTIC_ITEM,
    Warn,
    "repeated use of `MaybeDef::is_diag_item`/`TyCtxt::is_diagnostic_item`"
}
declare_lint_pass!(RepeatedIsDiagnosticItem => [REPEATED_IS_DIAGNOSTIC_ITEM]);

const NOTE: &str = "each call performs the same compiler query -- it's faster to query once, and reuse the results";

impl<'tcx> LateLintPass<'tcx> for RepeatedIsDiagnosticItem {
    #[expect(clippy::too_many_lines)]
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'_>) {
        for [(cond1, stmt1_span), (cond2, stmt2_span)] in block
            .stmts
            .windows(2)
            .filter_map(|pair| {
                if let [if1, if2] = pair
                    && let StmtKind::Expr(e1) | StmtKind::Semi(e1) = if1.kind
                    && let ExprKind::If(cond1, ..) = e1.kind
                    && let StmtKind::Expr(e2) | StmtKind::Semi(e2) = if2.kind
                    && let ExprKind::If(cond2, ..) = e2.kind
                {
                    Some([(cond1, if1.span), (cond2, if2.span)])
                } else {
                    None
                }
            })
            .chain(
                if let Some(if1) = block.stmts.last()
                    && let StmtKind::Expr(e1) | StmtKind::Semi(e1) = if1.kind
                    && let ExprKind::If(cond1, ..) = e1.kind
                    && let Some(e2) = block.expr
                    && let ExprKind::If(cond2, ..) = e2.kind
                {
                    Some([(cond1, if1.span), (cond2, e2.span)])
                } else {
                    None
                },
            )
        {
            let lint_span = stmt1_span.to(stmt2_span);

            // if recv1.is_diag_item(cx, sym1) && .. {
            //     ..
            // }
            // if recv2.is_diag_item(cx, sym2) && .. {
            //     ..
            // }
            if let Some(first @ (span1, (cx1, recv1, _))) = extract_nested_is_diag_item(cx, cond1)
                && let Some(second @ (span2, (cx2, recv2, _))) = extract_nested_is_diag_item(cx, cond2)
                && eq_expr_value(cx, cx1, cx2)
                && eq_expr_value(cx, recv1, recv2)
            {
                let recv_ty =
                    with_forced_trimmed_paths!(format!("{}", cx.typeck_results().expr_ty_adjusted(recv1).peel_refs()));
                let recv_ty = recv_ty.trim_end_matches("<'_>");
                span_lint_and_then(
                    cx,
                    REPEATED_IS_DIAGNOSTIC_ITEM,
                    lint_span,
                    format!("repeated calls to `{recv_ty}::is_diag_item`"),
                    |diag| {
                        diag.span_labels([span1, span2], "called here");
                        diag.note(NOTE);

                        let mut app = Applicability::HasPlaceholders;
                        let cx_str = snippet_with_applicability(cx, cx1.span, "_", &mut app);
                        let recv = snippet_with_applicability(cx, recv1.span, "_", &mut app);
                        let indent = snippet_indent(cx, stmt1_span).unwrap_or_default();
                        let sugg: Vec<_> = iter::once((
                            stmt1_span.shrink_to_lo(),
                            format!("let /* name */ = {recv}.opt_diag_name({cx_str});\n{indent}"),
                        )) // call `opt_diag_name` once
                        .chain([first, second].into_iter().map(|(expr_span, (_, _, sym))| {
                            let sym = snippet_with_applicability(cx, sym.span, "_", &mut app);
                            (expr_span, format!("/* name */ == Some({sym})"))
                        }))
                        .collect();

                        diag.multipart_suggestion_verbose(
                            format!("call `{recv_ty}::opt_diag_name`, and reuse the results"),
                            sugg,
                            app,
                        );
                    },
                );
                return;
            }

            // if cx.tcx.is_diagnostic_item(sym1, did) && .. {
            //     ..
            // }
            // if cx.tcx.is_diagnostic_item(sym2, did) && .. {
            //     ..
            // }
            if let Some(first @ (span1, (tcx1, did1, _))) = extract_nested_is_diagnostic_item(cx, cond1)
                && let Some(second @ (span2, (tcx2, did2, _))) = extract_nested_is_diagnostic_item(cx, cond2)
                && eq_expr_value(cx, tcx1, tcx2)
                && eq_expr_value(cx, did1, did2)
            {
                span_lint_and_then(
                    cx,
                    REPEATED_IS_DIAGNOSTIC_ITEM,
                    lint_span,
                    "repeated calls to `TyCtxt::is_diagnostic_item`",
                    |diag| {
                        diag.span_labels([span1, span2], "called here");
                        diag.note(NOTE);

                        let mut app = Applicability::HasPlaceholders;
                        let tcx = snippet_with_applicability(cx, tcx1.span, "_", &mut app);
                        let did = snippet_with_applicability(cx, did1.span, "_", &mut app);
                        let indent = snippet_indent(cx, stmt1_span).unwrap_or_default();
                        let sugg: Vec<_> = iter::once((
                            stmt1_span.shrink_to_lo(),
                            format!("let /* name */ = {tcx}.get_diagnostic_name({did});\n{indent}"),
                        )) // call `get_diagnostic_name` once
                        .chain([first, second].into_iter().map(|(expr_span, (_, _, sym))| {
                            let sym = snippet_with_applicability(cx, sym.span, "_", &mut app);
                            (expr_span, format!("/* name */ == Some({sym})"))
                        }))
                        .collect();

                        diag.multipart_suggestion_verbose(
                            "call `TyCtxt::get_diagnostic_name`, and reuse the results",
                            sugg,
                            app,
                        );
                    },
                );
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Binary(op, left, right) = expr.kind {
            if op.node == BinOpKind::Or {
                check_ors(cx, expr.span, left, right);
            } else if op.node == BinOpKind::And
                && let ExprKind::Unary(UnOp::Not, left) = left.kind
                && let ExprKind::Unary(UnOp::Not, right) = right.kind
            {
                check_ands(cx, expr.span, left, right);
            }
        } else if let (conds, _) = if_sequence(expr)
            && !conds.is_empty()
        {
            check_if_chains(cx, expr, conds);
        }
    }
}

fn check_ors(cx: &LateContext<'_>, span: Span, left: &Expr<'_>, right: &Expr<'_>) {
    // recv1.is_diag_item(cx, sym1) || recv2.is_diag_item(cx, sym2)
    if let Some((cx1, recv1, sym1)) = extract_is_diag_item(cx, left)
        && let Some((cx2, recv2, sym2)) = extract_is_diag_item(cx, right)
        && eq_expr_value(cx, cx1, cx2)
        && eq_expr_value(cx, recv1, recv2)
    {
        let recv_ty =
            with_forced_trimmed_paths!(format!("{}", cx.typeck_results().expr_ty_adjusted(recv1).peel_refs()));
        let recv_ty = recv_ty.trim_end_matches("<'_>");
        span_lint_and_then(
            cx,
            REPEATED_IS_DIAGNOSTIC_ITEM,
            span,
            format!("repeated calls to `{recv_ty}::is_diag_item`"),
            |diag| {
                diag.note(NOTE);

                let mut app = Applicability::MachineApplicable;
                let cx_str = snippet_with_applicability(cx, cx1.span, "_", &mut app);
                let recv = snippet_with_applicability(cx, recv1.span, "_", &mut app);
                let sym1 = snippet_with_applicability(cx, sym1.span, "_", &mut app);
                let sym2 = snippet_with_applicability(cx, sym2.span, "_", &mut app);
                diag.span_suggestion_verbose(
                    span,
                    format!("call `{recv_ty}::opt_diag_name`, and reuse the results"),
                    format!("matches!({recv}.opt_diag_name({cx_str}), Some({sym1} | {sym2}))"),
                    app,
                );
            },
        );
        return;
    }

    // cx.tcx.is_diagnostic_item(sym1, did) || cx.tcx.is_diagnostic_item(sym2, did)
    if let Some((tcx1, did1, sym1)) = extract_is_diagnostic_item(cx, left)
        && let Some((tcx2, did2, sym2)) = extract_is_diagnostic_item(cx, right)
        && eq_expr_value(cx, tcx1, tcx2)
        && eq_expr_value(cx, did1, did2)
    {
        span_lint_and_then(
            cx,
            REPEATED_IS_DIAGNOSTIC_ITEM,
            span,
            "repeated calls to `TyCtxt::is_diagnostic_item`",
            |diag| {
                diag.note(NOTE);

                let mut app = Applicability::MachineApplicable;
                let tcx = snippet_with_applicability(cx, tcx1.span, "_", &mut app);
                let did = snippet_with_applicability(cx, did1.span, "_", &mut app);
                let sym1 = snippet_with_applicability(cx, sym1.span, "_", &mut app);
                let sym2 = snippet_with_applicability(cx, sym2.span, "_", &mut app);
                diag.span_suggestion_verbose(
                    span,
                    "call `TyCtxt::get_diagnostic_name`, and reuse the results",
                    format!("matches!({tcx}.get_diagnostic_name({did}), Some({sym1} | {sym2}))"),
                    app,
                );
            },
        );
    }
}

fn check_ands(cx: &LateContext<'_>, span: Span, left: &Expr<'_>, right: &Expr<'_>) {
    // !recv1.is_diag_item(cx, sym1) && !recv2.is_diag_item(cx, sym2)
    if let Some((cx1, recv1, sym1)) = extract_is_diag_item(cx, left)
        && let Some((cx2, recv2, sym2)) = extract_is_diag_item(cx, right)
        && eq_expr_value(cx, cx1, cx2)
        && eq_expr_value(cx, recv1, recv2)
    {
        let recv_ty =
            with_forced_trimmed_paths!(format!("{}", cx.typeck_results().expr_ty_adjusted(recv1).peel_refs()));
        let recv_ty = recv_ty.trim_end_matches("<'_>");
        span_lint_and_then(
            cx,
            REPEATED_IS_DIAGNOSTIC_ITEM,
            span,
            format!("repeated calls to `{recv_ty}::is_diag_item`"),
            |diag| {
                diag.note(NOTE);

                let mut app = Applicability::MachineApplicable;
                let cx_str = snippet_with_applicability(cx, cx1.span, "_", &mut app);
                let recv = snippet_with_applicability(cx, recv1.span, "_", &mut app);
                let sym1 = snippet_with_applicability(cx, sym1.span, "_", &mut app);
                let sym2 = snippet_with_applicability(cx, sym2.span, "_", &mut app);
                diag.span_suggestion_verbose(
                    span,
                    format!("call `{recv_ty}::opt_diag_name`, and reuse the results"),
                    format!("!matches!({recv}.opt_diag_name({cx_str}), Some({sym1} | {sym2}))"),
                    app,
                );
            },
        );
        return;
    }

    // !cx.tcx.is_diagnostic_item(sym1, did) && !cx.tcx.is_diagnostic_item(sym2, did)
    if let Some((tcx1, did1, sym1)) = extract_is_diagnostic_item(cx, left)
        && let Some((tcx2, did2, sym2)) = extract_is_diagnostic_item(cx, right)
        && eq_expr_value(cx, tcx1, tcx2)
        && eq_expr_value(cx, did1, did2)
    {
        span_lint_and_then(
            cx,
            REPEATED_IS_DIAGNOSTIC_ITEM,
            span,
            "repeated calls to `TyCtxt::is_diagnostic_item`",
            |diag| {
                diag.note(NOTE);

                let mut app = Applicability::MachineApplicable;
                let tcx = snippet_with_applicability(cx, tcx1.span, "_", &mut app);
                let did = snippet_with_applicability(cx, did1.span, "_", &mut app);
                let sym1 = snippet_with_applicability(cx, sym1.span, "_", &mut app);
                let sym2 = snippet_with_applicability(cx, sym2.span, "_", &mut app);
                diag.span_suggestion_verbose(
                    span,
                    "call `TyCtxt::get_diagnostic_name`, and reuse the results",
                    format!("!matches!({tcx}.get_diagnostic_name({did}), Some({sym1} | {sym2}))"),
                    app,
                );
            },
        );
    }
}

fn check_if_chains<'tcx>(cx: &LateContext<'tcx>, expr: &Expr<'_>, conds: Vec<&'tcx Expr<'_>>) {
    // if ty.is_diag_item(cx, sym1) {
    //     ..
    // } else if ty.is_diag_item(cx, sym2) {
    //     ..
    // } else {
    //     ..
    // }
    let mut found = conds.iter().filter_map(|cond| extract_nested_is_diag_item(cx, cond));
    if let Some(first @ (_, (cx_1, recv1, _))) = found.next()
        && let other =
            found.filter(|(_, (cx_, recv, _))| eq_expr_value(cx, cx_, cx_1) && eq_expr_value(cx, recv, recv1))
        && let results = iter::once(first).chain(other).collect::<Vec<_>>()
        && results.len() > 1
    {
        let recv_ty =
            with_forced_trimmed_paths!(format!("{}", cx.typeck_results().expr_ty_adjusted(recv1).peel_refs()));
        let recv_ty = recv_ty.trim_end_matches("<'_>");
        span_lint_and_then(
            cx,
            REPEATED_IS_DIAGNOSTIC_ITEM,
            expr.span,
            format!("repeated calls to `{recv_ty}::is_diag_item`"),
            |diag| {
                diag.span_labels(results.iter().map(|(span, _)| *span), "called here");
                diag.note(NOTE);

                let mut app = Applicability::HasPlaceholders;
                let cx_str = snippet_with_applicability(cx, cx_1.span, "_", &mut app);
                let recv = snippet_with_applicability(cx, recv1.span, "_", &mut app);
                let span_before = if let Node::LetStmt(let_stmt) = cx.tcx.parent_hir_node(expr.hir_id) {
                    let_stmt.span
                } else {
                    expr.span
                };
                let indent = snippet_indent(cx, span_before).unwrap_or_default();
                let sugg: Vec<_> = iter::once((
                    span_before.shrink_to_lo(),
                    format!("let /* name */ = {recv}.opt_diag_name({cx_str});\n{indent}"),
                )) // call `opt_diag_name` once
                .chain(results.into_iter().map(|(expr_span, (_, _, sym))| {
                    let sym = snippet_with_applicability(cx, sym.span, "_", &mut app);
                    (expr_span, format!("/* name */ == Some({sym})"))
                }))
                .collect();

                diag.multipart_suggestion_verbose(
                    format!("call `{recv_ty}::opt_diag_name`, and reuse the results"),
                    sugg,
                    app,
                );
            },
        );
    }

    // if cx.tcx.is_diagnostic_item(sym1, did) {
    //     ..
    // } else if cx.tcx.is_diagnostic_item(sym2, did) {
    //     ..
    // } else {
    //     ..
    // }
    let mut found = conds
        .into_iter()
        .filter_map(|cond| extract_nested_is_diagnostic_item(cx, cond));
    if let Some(first @ (_, (tcx1, did1, _))) = found.next()
        && let other = found.filter(|(_, (tcx, did, _))| eq_expr_value(cx, tcx, tcx1) && eq_expr_value(cx, did, did1))
        && let results = iter::once(first).chain(other).collect::<Vec<_>>()
        && results.len() > 1
    {
        span_lint_and_then(
            cx,
            REPEATED_IS_DIAGNOSTIC_ITEM,
            expr.span,
            "repeated calls to `TyCtxt::is_diagnostic_item`",
            |diag| {
                diag.span_labels(results.iter().map(|(span, _)| *span), "called here");
                diag.note(NOTE);

                let mut app = Applicability::HasPlaceholders;
                let tcx = snippet_with_applicability(cx, tcx1.span, "_", &mut app);
                let recv = snippet_with_applicability(cx, did1.span, "_", &mut app);
                let span_before = if let Node::LetStmt(let_stmt) = cx.tcx.parent_hir_node(expr.hir_id) {
                    let_stmt.span
                } else {
                    expr.span
                };
                let indent = snippet_indent(cx, span_before).unwrap_or_default();
                let sugg: Vec<_> = iter::once((
                    span_before.shrink_to_lo(),
                    format!("let /* name */ = {tcx}.get_diagnostic_name({recv});\n{indent}"),
                )) // call `get_diagnostic_name` once
                .chain(results.into_iter().map(|(expr_span, (_, _, sym))| {
                    let sym = snippet_with_applicability(cx, sym.span, "_", &mut app);
                    (expr_span, format!("/* name */ == Some({sym})"))
                }))
                .collect();

                diag.multipart_suggestion_verbose(
                    "call `TyCtxt::get_diagnostic_name`, and reuse the results",
                    sugg,
                    app,
                );
            },
        );
    }
}

fn extract_is_diag_item<'tcx>(
    cx: &LateContext<'_>,
    expr: &'tcx Expr<'tcx>,
) -> Option<(&'tcx Expr<'tcx>, &'tcx Expr<'tcx>, &'tcx Expr<'tcx>)> {
    if let ExprKind::MethodCall(is_diag_item, recv, [cx_, sym], _) = expr.kind
        && is_diag_item.ident.name == sym::is_diag_item
        // Whether this a method from the `MaybeDef` trait
        && let Some(did) = cx.ty_based_def(expr).opt_parent(cx).opt_def_id()
        && MAYBE_DEF.matches(cx, did)
    {
        Some((cx_, recv, sym))
    } else {
        None
    }
}

fn extract_is_diagnostic_item<'tcx>(
    cx: &LateContext<'_>,
    expr: &'tcx Expr<'tcx>,
) -> Option<(&'tcx Expr<'tcx>, &'tcx Expr<'tcx>, &'tcx Expr<'tcx>)> {
    if let ExprKind::MethodCall(is_diag_item, tcx, [sym, did], _) = expr.kind
        && is_diag_item.ident.name == sym::is_diagnostic_item
        // Whether this is an inherent method on `TyCtxt`
        && cx
            .ty_based_def(expr)
            .opt_parent(cx)
            .opt_impl_ty(cx)
            .is_diag_item(cx, sym::TyCtxt)
    {
        Some((tcx, did, sym))
    } else {
        None
    }
}

fn extract_nested_is_diag_item<'tcx>(
    cx: &LateContext<'tcx>,
    cond: &'tcx Expr<'_>,
) -> Option<(Span, (&'tcx Expr<'tcx>, &'tcx Expr<'tcx>, &'tcx Expr<'tcx>))> {
    for_each_expr(cx, cond, |cond_part| {
        if let Some(res) = extract_is_diag_item(cx, cond_part) {
            ControlFlow::Break((cond_part.span, res))
        } else {
            ControlFlow::Continue(())
        }
    })
}

fn extract_nested_is_diagnostic_item<'tcx>(
    cx: &LateContext<'tcx>,
    cond: &'tcx Expr<'_>,
) -> Option<(Span, (&'tcx Expr<'tcx>, &'tcx Expr<'tcx>, &'tcx Expr<'tcx>))> {
    for_each_expr(cx, cond, |cond_part| {
        if let Some(res) = extract_is_diagnostic_item(cx, cond_part) {
            ControlFlow::Break((cond_part.span, res))
        } else {
            ControlFlow::Continue(())
        }
    })
}
