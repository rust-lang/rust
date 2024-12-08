use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::path_res;
use clippy_utils::source::snippet;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Block, Body, Expr, ExprKind, LangItem, MatchSource, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Suggests alternatives for useless applications of `?` in terminating expressions
    ///
    /// ### Why is this bad?
    /// There's no reason to use `?` to short-circuit when execution of the body will end there anyway.
    ///
    /// ### Example
    /// ```no_run
    /// struct TO {
    ///     magic: Option<usize>,
    /// }
    ///
    /// fn f(to: TO) -> Option<usize> {
    ///     Some(to.magic?)
    /// }
    ///
    /// struct TR {
    ///     magic: Result<usize, bool>,
    /// }
    ///
    /// fn g(tr: Result<TR, bool>) -> Result<usize, bool> {
    ///     tr.and_then(|t| Ok(t.magic?))
    /// }
    ///
    /// ```
    /// Use instead:
    /// ```no_run
    /// struct TO {
    ///     magic: Option<usize>,
    /// }
    ///
    /// fn f(to: TO) -> Option<usize> {
    ///    to.magic
    /// }
    ///
    /// struct TR {
    ///     magic: Result<usize, bool>,
    /// }
    ///
    /// fn g(tr: Result<TR, bool>) -> Result<usize, bool> {
    ///     tr.and_then(|t| t.magic)
    /// }
    /// ```
    #[clippy::version = "1.51.0"]
    pub NEEDLESS_QUESTION_MARK,
    complexity,
    "Suggest `value.inner_option` instead of `Some(value.inner_option?)`. The same goes for `Result<T, E>`."
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
        && let sugg_remove = if cx.tcx.lang_items().option_some_variant() == Some(variant_id) {
            "Some()"
        } else if cx.tcx.lang_items().result_ok_variant() == Some(variant_id) {
            "Ok()"
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
        span_lint_and_sugg(
            cx,
            NEEDLESS_QUESTION_MARK,
            expr.span,
            "question mark operator is useless here",
            format!("try removing question mark and `{sugg_remove}`"),
            format!("{}", snippet(cx, inner_expr.span, r#""...""#)),
            Applicability::MachineApplicable,
        );
    }
}
