use crate::utils::{in_constant, match_qpath, match_trait_method, paths, snippet, span_lint_and_then};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Arm, Expr, ExprKind, HirId, MatchSource, PatKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_mir::const_eval::is_const_fn;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Symbol;

declare_clippy_lint! {
    /// **What it does:** Lint for redundant pattern matching over `Result` or
    /// `Option`
    ///
    /// **Why is this bad?** It's more concise and clear to just use the proper
    /// utility function
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// if let Ok(_) = Ok::<i32, i32>(42) {}
    /// if let Err(_) = Err::<i32, i32>(42) {}
    /// if let None = None::<()> {}
    /// if let Some(_) = Some(42) {}
    /// match Ok::<i32, i32>(42) {
    ///     Ok(_) => true,
    ///     Err(_) => false,
    /// };
    /// ```
    ///
    /// The more idiomatic use would be:
    ///
    /// ```rust
    /// if Ok::<i32, i32>(42).is_ok() {}
    /// if Err::<i32, i32>(42).is_err() {}
    /// if None::<()>.is_none() {}
    /// if Some(42).is_some() {}
    /// Ok::<i32, i32>(42).is_ok();
    /// ```
    pub REDUNDANT_PATTERN_MATCHING,
    style,
    "use the proper utility function avoiding an `if let`"
}

declare_lint_pass!(RedundantPatternMatching => [REDUNDANT_PATTERN_MATCHING]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for RedundantPatternMatching {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Match(op, arms, ref match_source) = &expr.kind {
            match match_source {
                MatchSource::Normal => find_sugg_for_match(cx, expr, op, arms),
                MatchSource::IfLetDesugar { .. } => find_sugg_for_if_let(cx, expr, op, arms, "if"),
                MatchSource::WhileLetDesugar => find_sugg_for_if_let(cx, expr, op, arms, "while"),
                _ => return,
            }
        }
    }
}

fn find_sugg_for_if_let<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    expr: &'tcx Expr<'_>,
    op: &Expr<'_>,
    arms: &[Arm<'_>],
    keyword: &'static str,
) {
    fn find_suggestion(cx: &LateContext<'_, '_>, hir_id: HirId, path: &QPath<'_>) -> Option<&'static str> {
        if match_qpath(path, &paths::RESULT_OK) && can_suggest(cx, hir_id, sym!(result_type), "is_ok") {
            return Some("is_ok()");
        }
        if match_qpath(path, &paths::RESULT_ERR) && can_suggest(cx, hir_id, sym!(result_type), "is_err") {
            return Some("is_err()");
        }
        if match_qpath(path, &paths::OPTION_SOME) && can_suggest(cx, hir_id, sym!(option_type), "is_some") {
            return Some("is_some()");
        }
        if match_qpath(path, &paths::OPTION_NONE) && can_suggest(cx, hir_id, sym!(option_type), "is_none") {
            return Some("is_none()");
        }
        None
    }

    let hir_id = expr.hir_id;
    let good_method = match arms[0].pat.kind {
        PatKind::TupleStruct(ref path, ref patterns, _) if patterns.len() == 1 => {
            if let PatKind::Wild = patterns[0].kind {
                find_suggestion(cx, hir_id, path)
            } else {
                None
            }
        },
        PatKind::Path(ref path) => find_suggestion(cx, hir_id, path),
        _ => None,
    };
    let good_method = match good_method {
        Some(method) => method,
        None => return,
    };

    // check that `while_let_on_iterator` lint does not trigger
    if_chain! {
        if keyword == "while";
        if let ExprKind::MethodCall(method_path, _, _, _) = op.kind;
        if method_path.ident.name == sym!(next);
        if match_trait_method(cx, op, &paths::ITERATOR);
        then {
            return;
        }
    }

    span_lint_and_then(
        cx,
        REDUNDANT_PATTERN_MATCHING,
        arms[0].pat.span,
        &format!("redundant pattern matching, consider using `{}`", good_method),
        |diag| {
            // while let ... = ... { ... }
            // ^^^^^^^^^^^^^^^^^^^^^^^^^^^
            let expr_span = expr.span;

            // while let ... = ... { ... }
            //                 ^^^
            let op_span = op.span.source_callsite();

            // while let ... = ... { ... }
            // ^^^^^^^^^^^^^^^^^^^
            let span = expr_span.until(op_span.shrink_to_hi());
            diag.span_suggestion(
                span,
                "try this",
                format!("{} {}.{}", keyword, snippet(cx, op_span, "_"), good_method),
                Applicability::MachineApplicable, // snippet
            );
        },
    );
}

fn find_sugg_for_match<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>, op: &Expr<'_>, arms: &[Arm<'_>]) {
    if arms.len() == 2 {
        let node_pair = (&arms[0].pat.kind, &arms[1].pat.kind);

        let hir_id = expr.hir_id;
        let found_good_method = match node_pair {
            (
                PatKind::TupleStruct(ref path_left, ref patterns_left, _),
                PatKind::TupleStruct(ref path_right, ref patterns_right, _),
            ) if patterns_left.len() == 1 && patterns_right.len() == 1 => {
                if let (PatKind::Wild, PatKind::Wild) = (&patterns_left[0].kind, &patterns_right[0].kind) {
                    find_good_method_for_match(
                        arms,
                        path_left,
                        path_right,
                        &paths::RESULT_OK,
                        &paths::RESULT_ERR,
                        "is_ok()",
                        "is_err()",
                        || can_suggest(cx, hir_id, sym!(result_type), "is_ok"),
                        || can_suggest(cx, hir_id, sym!(result_type), "is_err"),
                    )
                } else {
                    None
                }
            },
            (PatKind::TupleStruct(ref path_left, ref patterns, _), PatKind::Path(ref path_right))
            | (PatKind::Path(ref path_left), PatKind::TupleStruct(ref path_right, ref patterns, _))
                if patterns.len() == 1 =>
            {
                if let PatKind::Wild = patterns[0].kind {
                    find_good_method_for_match(
                        arms,
                        path_left,
                        path_right,
                        &paths::OPTION_SOME,
                        &paths::OPTION_NONE,
                        "is_some()",
                        "is_none()",
                        || can_suggest(cx, hir_id, sym!(option_type), "is_some"),
                        || can_suggest(cx, hir_id, sym!(option_type), "is_none"),
                    )
                } else {
                    None
                }
            },
            _ => None,
        };

        if let Some(good_method) = found_good_method {
            span_lint_and_then(
                cx,
                REDUNDANT_PATTERN_MATCHING,
                expr.span,
                &format!("redundant pattern matching, consider using `{}`", good_method),
                |diag| {
                    let span = expr.span.to(op.span);
                    diag.span_suggestion(
                        span,
                        "try this",
                        format!("{}.{}", snippet(cx, op.span, "_"), good_method),
                        Applicability::MaybeIncorrect, // snippet
                    );
                },
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn find_good_method_for_match<'a>(
    arms: &[Arm<'_>],
    path_left: &QPath<'_>,
    path_right: &QPath<'_>,
    expected_left: &[&str],
    expected_right: &[&str],
    should_be_left: &'a str,
    should_be_right: &'a str,
    can_suggest_left: impl Fn() -> bool,
    can_suggest_right: impl Fn() -> bool,
) -> Option<&'a str> {
    let body_node_pair = if match_qpath(path_left, expected_left) && match_qpath(path_right, expected_right) {
        (&(*arms[0].body).kind, &(*arms[1].body).kind)
    } else if match_qpath(path_right, expected_left) && match_qpath(path_left, expected_right) {
        (&(*arms[1].body).kind, &(*arms[0].body).kind)
    } else {
        return None;
    };

    match body_node_pair {
        (ExprKind::Lit(ref lit_left), ExprKind::Lit(ref lit_right)) => match (&lit_left.node, &lit_right.node) {
            (LitKind::Bool(true), LitKind::Bool(false)) if can_suggest_left() => Some(should_be_left),
            (LitKind::Bool(false), LitKind::Bool(true)) if can_suggest_right() => Some(should_be_right),
            _ => None,
        },
        _ => None,
    }
}

fn can_suggest(cx: &LateContext<'_, '_>, hir_id: HirId, diag_item: Symbol, name: &str) -> bool {
    if !in_constant(cx, hir_id) {
        return true;
    }

    // Avoid suggesting calls to non-`const fn`s in const contexts, see #5697.
    cx.tcx
        .get_diagnostic_item(diag_item)
        .and_then(|def_id| {
            cx.tcx.inherent_impls(def_id).iter().find_map(|imp| {
                cx.tcx
                    .associated_items(*imp)
                    .in_definition_order()
                    .find_map(|item| match item.kind {
                        ty::AssocKind::Fn if item.ident.name.as_str() == name => Some(item.def_id),
                        _ => None,
                    })
            })
        })
        .map_or(false, |def_id| is_const_fn(cx.tcx, def_id))
}
