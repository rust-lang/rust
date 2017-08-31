use rustc::lint::*;
use rustc::hir::*;
use syntax::codemap::Span;
use utils::{paths, span_lint_and_then, match_qpath, snippet};

/// **What it does:*** Lint for redundant pattern matching over `Result` or
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
/// ```
///
/// The more idiomatic use would be:
///
/// ```rust
/// if Ok::<i32, i32>(42).is_ok() {}
/// if Err::<i32, i32>(42).is_err() {}
/// if None::<()>.is_none() {}
/// if Some(42).is_some() {}
/// ```
///
declare_lint! {
    pub IF_LET_REDUNDANT_PATTERN_MATCHING,
    Warn,
    "use the proper utility function avoiding an `if let`"
}

#[derive(Copy, Clone)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(IF_LET_REDUNDANT_PATTERN_MATCHING)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {

        if let ExprMatch(ref op, ref arms, MatchSource::IfLetDesugar { .. }) = expr.node {

            if arms[0].pats.len() == 1 {

                let good_method = match arms[0].pats[0].node {
                    PatKind::TupleStruct(ref path, ref pats, _) if pats.len() == 1 && pats[0].node == PatKind::Wild => {
                        if match_qpath(path, &paths::RESULT_OK) {
                            "is_ok()"
                        } else if match_qpath(path, &paths::RESULT_ERR) {
                            "is_err()"
                        } else if match_qpath(path, &paths::OPTION_SOME) {
                            "is_some()"
                        } else {
                            return;
                        }
                    },

                    PatKind::Path(ref path) if match_qpath(path, &paths::OPTION_NONE) => "is_none()",

                    _ => return,
                };

                span_lint_and_then(cx,
                                   IF_LET_REDUNDANT_PATTERN_MATCHING,
                                   arms[0].pats[0].span,
                                   &format!("redundant pattern matching, consider using `{}`", good_method),
                                   |db| {
                    let span = Span::new(
                        expr.span.lo(),
                        op.span.hi(),
                        expr.span.ctxt(),
                    );
                    db.span_suggestion(span, "try this", format!("if {}.{}", snippet(cx, op.span, "_"), good_method));
                });
            }

        }
    }
}
