use clippy_utils::consts::{constant, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::VecArgs;
use clippy_utils::macros::root_macro_call;
use clippy_utils::source::snippet;
use clippy_utils::{expr_or_init, fn_def_id, match_def_path, paths};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::{sym, Span};

declare_clippy_lint! {
    /// ### What it does
    /// Looks for patterns such as `vec![Vec::with_capacity(x); n]` or `iter::repeat(Vec::with_capacity(x))`.
    ///
    /// ### Why is this bad?
    /// These constructs work by cloning the element, but cloning a `Vec<_>` does not
    /// respect the old vector's capacity and effectively discards it.
    ///
    /// This makes `iter::repeat(Vec::with_capacity(x))` especially suspicious because the user most certainly
    /// expected that the yielded `Vec<_>` will have the requested capacity, otherwise one can simply write
    /// `iter::repeat(Vec::new())` instead and it will have the same effect.
    ///
    /// Similarly for `vec![x; n]`, the element `x` is cloned to fill the vec.
    /// Unlike `iter::repeat` however, the vec repeat macro does not have to clone the value `n` times
    /// but just `n - 1` times, because it can reuse the passed value for the last slot.
    /// That means that the last `Vec<_>` gets the requested capacity but all other ones do not.
    ///
    /// ### Example
    /// ```rust
    /// # use std::iter;
    ///
    /// let _: Vec<Vec<u8>> = vec![Vec::with_capacity(42); 123];
    /// let _: Vec<Vec<u8>> = iter::repeat(Vec::with_capacity(42)).take(123).collect();
    /// ```
    /// Use instead:
    /// ```rust
    /// # use std::iter;
    ///
    /// let _: Vec<Vec<u8>> = iter::repeat_with(|| Vec::with_capacity(42)).take(123).collect();
    /// //                                      ^^^ this closure executes 123 times
    /// //                                          and the vecs will have the expected capacity
    /// ```
    #[clippy::version = "1.76.0"]
    pub REPEAT_VEC_WITH_CAPACITY,
    suspicious,
    "repeating a `Vec::with_capacity` expression which does not retain capacity"
}

declare_lint_pass!(RepeatVecWithCapacity => [REPEAT_VEC_WITH_CAPACITY]);

fn emit_lint(cx: &LateContext<'_>, span: Span, kind: &str, note: &'static str, sugg_msg: &'static str, sugg: String) {
    span_lint_and_then(
        cx,
        REPEAT_VEC_WITH_CAPACITY,
        span,
        &format!("repeating `Vec::with_capacity` using `{kind}`, which does not retain capacity"),
        |diag| {
            diag.note(note);
            diag.span_suggestion_verbose(span, sugg_msg, sugg, Applicability::MaybeIncorrect);
        },
    );
}

/// Checks `vec![Vec::with_capacity(x); n]`
fn check_vec_macro(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if let Some(mac_call) = root_macro_call(expr.span)
        && cx.tcx.is_diagnostic_item(sym::vec_macro, mac_call.def_id)
        && let Some(VecArgs::Repeat(repeat_expr, len_expr)) = VecArgs::hir(cx, expr)
        && fn_def_id(cx, repeat_expr).is_some_and(|did| match_def_path(cx, did, &paths::VEC_WITH_CAPACITY))
        && !len_expr.span.from_expansion()
        && let Some(Constant::Int(2..)) = constant(cx, cx.typeck_results(), expr_or_init(cx, len_expr))
    {
        emit_lint(
            cx,
            expr.span.source_callsite(),
            "vec![x; n]",
            "only the last `Vec` will have the capacity",
            "if you intended to initialize multiple `Vec`s with an initial capacity, try",
            format!(
                "(0..{}).map(|_| {}).collect::<Vec<_>>()",
                snippet(cx, len_expr.span, ""),
                snippet(cx, repeat_expr.span, "..")
            ),
        );
    }
}

/// Checks `iter::repeat(Vec::with_capacity(x))`
fn check_repeat_fn(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if !expr.span.from_expansion()
        && fn_def_id(cx, expr).is_some_and(|did| cx.tcx.is_diagnostic_item(sym::iter_repeat, did))
        && let ExprKind::Call(_, [repeat_expr]) = expr.kind
        && fn_def_id(cx, repeat_expr).is_some_and(|did| match_def_path(cx, did, &paths::VEC_WITH_CAPACITY))
        && !repeat_expr.span.from_expansion()
    {
        emit_lint(
            cx,
            expr.span,
            "iter::repeat",
            "none of the yielded `Vec`s will have the requested capacity",
            "if you intended to create an iterator that yields `Vec`s with an initial capacity, try",
            format!("std::iter::repeat_with(|| {})", snippet(cx, repeat_expr.span, "..")),
        );
    }
}

impl LateLintPass<'_> for RepeatVecWithCapacity {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        check_vec_macro(cx, expr);
        check_repeat_fn(cx, expr);
    }
}
