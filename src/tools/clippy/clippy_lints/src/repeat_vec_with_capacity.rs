use clippy_config::Conf;
use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::higher::VecArgs;
use clippy_utils::macros::matching_root_macro_call;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet;
use clippy_utils::{expr_or_init, fn_def_id, std_or_core};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::{Span, sym};

pub struct RepeatVecWithCapacity {
    msrv: Msrv,
}

impl RepeatVecWithCapacity {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

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

impl_lint_pass!(RepeatVecWithCapacity => [REPEAT_VEC_WITH_CAPACITY]);

fn emit_lint(cx: &LateContext<'_>, span: Span, kind: &str, note: &'static str, sugg_msg: &'static str, sugg: String) {
    span_lint_and_then(
        cx,
        REPEAT_VEC_WITH_CAPACITY,
        span,
        format!("repeating `Vec::with_capacity` using `{kind}`, which does not retain capacity"),
        |diag| {
            diag.note(note);
            diag.span_suggestion_verbose(span, sugg_msg, sugg, Applicability::MaybeIncorrect);
        },
    );
}

/// Checks `vec![Vec::with_capacity(x); n]`
fn check_vec_macro(cx: &LateContext<'_>, expr: &Expr<'_>) {
    if matching_root_macro_call(cx, expr.span, sym::vec_macro).is_some()
        && let Some(VecArgs::Repeat(repeat_expr, len_expr)) = VecArgs::hir(cx, expr)
        && fn_def_id(cx, repeat_expr).is_some_and(|did| cx.tcx.is_diagnostic_item(sym::vec_with_capacity, did))
        && !len_expr.span.from_expansion()
        && let Some(Constant::Int(2..)) = ConstEvalCtxt::new(cx).eval(expr_or_init(cx, len_expr))
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
fn check_repeat_fn(cx: &LateContext<'_>, expr: &Expr<'_>, msrv: Msrv) {
    if !expr.span.from_expansion()
        && fn_def_id(cx, expr).is_some_and(|did| cx.tcx.is_diagnostic_item(sym::iter_repeat, did))
        && let ExprKind::Call(_, [repeat_expr]) = expr.kind
        && fn_def_id(cx, repeat_expr).is_some_and(|did| cx.tcx.is_diagnostic_item(sym::vec_with_capacity, did))
        && !repeat_expr.span.from_expansion()
        && let Some(exec_context) = std_or_core(cx)
        && msrv.meets(cx, msrvs::REPEAT_WITH)
    {
        emit_lint(
            cx,
            expr.span,
            "iter::repeat",
            "none of the yielded `Vec`s will have the requested capacity",
            "if you intended to create an iterator that yields `Vec`s with an initial capacity, try",
            format!(
                "{exec_context}::iter::repeat_with(|| {})",
                snippet(cx, repeat_expr.span, "..")
            ),
        );
    }
}

impl LateLintPass<'_> for RepeatVecWithCapacity {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        check_vec_macro(cx, expr);
        check_repeat_fn(cx, expr, self.msrv);
    }
}
