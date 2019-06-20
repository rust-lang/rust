use crate::utils::{match_qpath, paths, snippet, span_lint_and_then};
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::ast::LitKind;
use syntax::ptr::P;

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
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprKind::Match(ref op, ref arms, ref match_source) = expr.node {
            match match_source {
                MatchSource::Normal => find_sugg_for_match(cx, expr, op, arms),
                MatchSource::IfLetDesugar { .. } => find_sugg_for_if_let(cx, expr, op, arms),
                _ => return,
            }
        }
    }
}

fn find_sugg_for_if_let<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr, op: &P<Expr>, arms: &HirVec<Arm>) {
    if arms[0].pats.len() == 1 {
        let good_method = match arms[0].pats[0].node {
            PatKind::TupleStruct(ref path, ref patterns, _) if patterns.len() == 1 => {
                if let PatKind::Wild = patterns[0].node {
                    if match_qpath(path, &paths::RESULT_OK) {
                        "is_ok()"
                    } else if match_qpath(path, &paths::RESULT_ERR) {
                        "is_err()"
                    } else if match_qpath(path, &paths::OPTION_SOME) {
                        "is_some()"
                    } else {
                        return;
                    }
                } else {
                    return;
                }
            },

            PatKind::Path(ref path) if match_qpath(path, &paths::OPTION_NONE) => "is_none()",

            _ => return,
        };

        span_lint_and_then(
            cx,
            REDUNDANT_PATTERN_MATCHING,
            arms[0].pats[0].span,
            &format!("redundant pattern matching, consider using `{}`", good_method),
            |db| {
                let span = expr.span.to(op.span);
                db.span_suggestion(
                    span,
                    "try this",
                    format!("if {}.{}", snippet(cx, op.span, "_"), good_method),
                    Applicability::MachineApplicable, // snippet
                );
            },
        );
    }
}

fn find_sugg_for_match<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr, op: &P<Expr>, arms: &HirVec<Arm>) {
    if arms.len() == 2 {
        let node_pair = (&arms[0].pats[0].node, &arms[1].pats[0].node);

        let found_good_method = match node_pair {
            (
                PatKind::TupleStruct(ref path_left, ref patterns_left, _),
                PatKind::TupleStruct(ref path_right, ref patterns_right, _),
            ) if patterns_left.len() == 1 && patterns_right.len() == 1 => {
                if let (PatKind::Wild, PatKind::Wild) = (&patterns_left[0].node, &patterns_right[0].node) {
                    find_good_method_for_match(
                        arms,
                        path_left,
                        path_right,
                        &paths::RESULT_OK,
                        &paths::RESULT_ERR,
                        "is_ok()",
                        "is_err()",
                    )
                } else {
                    None
                }
            },
            (PatKind::TupleStruct(ref path_left, ref patterns, _), PatKind::Path(ref path_right))
            | (PatKind::Path(ref path_left), PatKind::TupleStruct(ref path_right, ref patterns, _))
                if patterns.len() == 1 =>
            {
                if let PatKind::Wild = patterns[0].node {
                    find_good_method_for_match(
                        arms,
                        path_left,
                        path_right,
                        &paths::OPTION_SOME,
                        &paths::OPTION_NONE,
                        "is_some()",
                        "is_none()",
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
                |db| {
                    let span = expr.span.to(op.span);
                    db.span_suggestion(
                        span,
                        "try this",
                        format!("{}.{}", snippet(cx, op.span, "_"), good_method),
                        Applicability::MachineApplicable, // snippet
                    );
                },
            );
        }
    }
}

fn find_good_method_for_match<'a>(
    arms: &HirVec<Arm>,
    path_left: &QPath,
    path_right: &QPath,
    expected_left: &[&str],
    expected_right: &[&str],
    should_be_left: &'a str,
    should_be_right: &'a str,
) -> Option<&'a str> {
    let body_node_pair = if match_qpath(path_left, expected_left) && match_qpath(path_right, expected_right) {
        (&(*arms[0].body).node, &(*arms[1].body).node)
    } else if match_qpath(path_right, expected_left) && match_qpath(path_left, expected_right) {
        (&(*arms[1].body).node, &(*arms[0].body).node)
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
