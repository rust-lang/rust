use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::eager_or_lazy::switch_to_eager_eval;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::{snippet_with_applicability, snippet_with_context, walk_span_to_context};
use clippy_utils::sugg::Sugg;
use clippy_utils::{
    contains_return, expr_adjustment_requires_coercion, higher, is_else_clause, is_in_const_context, is_res_lang_ctor,
    path_res, peel_blocks, sym,
};
use rustc_errors::Applicability;
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for if-else that could be written using either `bool::then` or `bool::then_some`.
    ///
    /// ### Why restrict this?
    /// Looks a little redundant. Using `bool::then` is more concise and incurs no loss of clarity.
    /// For simple calculations and known values, use `bool::then_some`, which is eagerly evaluated
    /// in comparison to `bool::then`.
    ///
    /// ### Example
    /// ```no_run
    /// # let v = vec![0];
    /// let a = if v.is_empty() {
    ///     println!("true!");
    ///     Some(42)
    /// } else {
    ///     None
    /// };
    /// ```
    ///
    /// Could be written:
    ///
    /// ```no_run
    /// # let v = vec![0];
    /// let a = v.is_empty().then(|| {
    ///     println!("true!");
    ///     42
    /// });
    /// ```
    #[clippy::version = "1.53.0"]
    pub IF_THEN_SOME_ELSE_NONE,
    restriction,
    "Finds if-else that could be written using either `bool::then` or `bool::then_some`"
}

pub struct IfThenSomeElseNone {
    msrv: Msrv,
}

impl IfThenSomeElseNone {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(IfThenSomeElseNone => [IF_THEN_SOME_ELSE_NONE]);

impl<'tcx> LateLintPass<'tcx> for IfThenSomeElseNone {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let Some(higher::If {
            cond,
            then,
            r#else: Some(els),
        }) = higher::If::hir(expr)
            && let ExprKind::Block(then_block, _) = then.kind
            && let Some(then_expr) = then_block.expr
            && let ExprKind::Call(then_call, [then_arg]) = then_expr.kind
            && !expr.span.from_expansion()
            && !then_expr.span.from_expansion()
            && is_res_lang_ctor(cx, path_res(cx, then_call), OptionSome)
            && is_res_lang_ctor(cx, path_res(cx, peel_blocks(els)), OptionNone)
            && !is_else_clause(cx.tcx, expr)
            && !is_in_const_context(cx)
            && self.msrv.meets(cx, msrvs::BOOL_THEN)
            && !contains_return(then_block.stmts)
        {
            let method_name = if switch_to_eager_eval(cx, expr) && self.msrv.meets(cx, msrvs::BOOL_THEN_SOME) {
                sym::then_some
            } else {
                sym::then
            };
            let ctxt = expr.span.ctxt();

            span_lint_and_then(
                cx,
                IF_THEN_SOME_ELSE_NONE,
                expr.span,
                format!("this could be simplified with `bool::{method_name}`"),
                |diag| {
                    if expr_adjustment_requires_coercion(cx, then_arg) {
                        return;
                    }

                    let mut app = Applicability::MachineApplicable;
                    let cond_snip = Sugg::hir_with_context(cx, cond, ctxt, "[condition]", &mut app)
                        .maybe_paren()
                        .to_string();
                    let arg_snip = snippet_with_context(cx, then_arg.span, ctxt, "[body]", &mut app).0;
                    let method_body = if let Some(first_stmt) = then_block.stmts.first()
                        && let Some(first_stmt_span) = walk_span_to_context(first_stmt.span, ctxt)
                    {
                        let block_snippet =
                            snippet_with_applicability(cx, first_stmt_span.until(then_expr.span), "..", &mut app);
                        let closure = if method_name == sym::then { "|| " } else { "" };
                        format!("{closure} {{ {} {arg_snip} }}", block_snippet.trim_end())
                    } else if method_name == sym::then {
                        (std::borrow::Cow::Borrowed("|| ") + arg_snip).into_owned()
                    } else {
                        arg_snip.into_owned()
                    };

                    diag.span_suggestion(
                        expr.span,
                        "try",
                        format!("{cond_snip}.{method_name}({method_body})"),
                        app,
                    );
                },
            );
        }
    }
}
