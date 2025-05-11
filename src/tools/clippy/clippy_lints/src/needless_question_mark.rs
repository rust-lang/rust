use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::path_res;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Block, Body, Expr, ExprKind, LangItem, MatchSource, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Suggests replacing `Ok(x?)` or `Some(x?)` with `x` in return positions where the `?` operator
    /// is not needed to convert the type of `x`.
    ///
    /// ### Why is this bad?
    /// There's no reason to use `?` to short-circuit when execution of the body will end there anyway.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::num::ParseIntError;
    /// fn f(s: &str) -> Option<usize> {
    ///     Some(s.find('x')?)
    /// }
    ///
    /// fn g(s: &str) -> Result<usize, ParseIntError> {
    ///     Ok(s.parse()?)
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use std::num::ParseIntError;
    /// fn f(s: &str) -> Option<usize> {
    ///     s.find('x')
    /// }
    ///
    /// fn g(s: &str) -> Result<usize, ParseIntError> {
    ///     s.parse()
    /// }
    /// ```
    #[clippy::version = "1.51.0"]
    pub NEEDLESS_QUESTION_MARK,
    complexity,
    "using `Ok(x?)` or `Some(x?)` where `x` would be equivalent"
}

declare_lint_pass!(NeedlessQuestionMark => [NEEDLESS_QUESTION_MARK]);

impl LateLintPass<'_> for NeedlessQuestionMark {
    /*
     * The question mark operator is compatible with both Result<T, E> and Option<T>,
     * from Rust 1.13 and 1.22 respectively.
     */

    /*
     * What do we match:
     * Expressions that look like this:
     * Some(option?), Ok(result?)
     *
     * Where do we match:
     *      Last expression of a body
     *      Return statement
     *      A body's value (single line closure)
     *
     * What do we not match:
     *      Implicit calls to `from(..)` on the error value
     */

    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &'_ Expr<'_>) {
        if let ExprKind::Ret(Some(e)) = expr.kind {
            check(cx, e);
        }
    }

    fn check_body(&mut self, cx: &LateContext<'_>, body: &'_ Body<'_>) {
        if let ExprKind::Block(
            Block {
                expr:
                    Some(Expr {
                        kind: ExprKind::DropTemps(async_body),
                        ..
                    }),
                ..
            },
            _,
        ) = body.value.kind
        {
            if let ExprKind::Block(Block { expr: Some(expr), .. }, ..) = async_body.kind {
                check(cx, expr.peel_blocks());
            }
        } else {
            check(cx, body.value.peel_blocks());
        }
    }
}

fn check(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let ExprKind::Call(path, [arg]) = expr.kind
        && let Res::Def(DefKind::Ctor(..), ctor_id) = path_res(cx, path)
        && let Some(variant_id) = cx.tcx.opt_parent(ctor_id)
        && let variant = if cx.tcx.lang_items().option_some_variant() == Some(variant_id) {
            "Some"
        } else if cx.tcx.lang_items().result_ok_variant() == Some(variant_id) {
            "Ok"
        } else {
            return;
        }
        && let ExprKind::Match(inner_expr_with_q, _, MatchSource::TryDesugar(_)) = &arg.kind
        && let ExprKind::Call(called, [inner_expr]) = &inner_expr_with_q.kind
        && let ExprKind::Path(QPath::LangItem(LangItem::TryTraitBranch, ..)) = &called.kind
        && expr.span.eq_ctxt(inner_expr.span)
        && let expr_ty = cx.typeck_results().expr_ty(expr)
        && let inner_ty = cx.typeck_results().expr_ty(inner_expr)
        && expr_ty == inner_ty
    {
        span_lint_hir_and_then(
            cx,
            NEEDLESS_QUESTION_MARK,
            expr.hir_id,
            expr.span,
            format!("enclosing `{variant}` and `?` operator are unneeded"),
            |diag| {
                diag.multipart_suggestion(
                    format!("remove the enclosing `{variant}` and `?` operator"),
                    vec![
                        (expr.span.until(inner_expr.span), String::new()),
                        (
                            inner_expr.span.shrink_to_hi().to(expr.span.shrink_to_hi()),
                            String::new(),
                        ),
                    ],
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}
