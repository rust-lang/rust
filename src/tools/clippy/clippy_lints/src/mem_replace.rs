use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::is_non_aggregate_primitive_type;
use clippy_utils::{
    is_default_equivalent, is_expr_used_or_unified, is_res_lang_ctor, path_res, peel_ref_operators, std_or_core,
};
use rustc_errors::Applicability;
use rustc_hir::LangItem::OptionNone;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::lint::in_external_macro;
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `mem::replace()` on an `Option` with
    /// `None`.
    ///
    /// ### Why is this bad?
    /// `Option` already has the method `take()` for
    /// taking its current value (Some(..) or None) and replacing it with
    /// `None`.
    ///
    /// ### Example
    /// ```no_run
    /// use std::mem;
    ///
    /// let mut an_option = Some(0);
    /// let replaced = mem::replace(&mut an_option, None);
    /// ```
    /// Is better expressed with:
    /// ```no_run
    /// let mut an_option = Some(0);
    /// let taken = an_option.take();
    /// ```
    #[clippy::version = "1.31.0"]
    pub MEM_REPLACE_OPTION_WITH_NONE,
    style,
    "replacing an `Option` with `None` instead of `take()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `mem::replace(&mut _, mem::uninitialized())`
    /// and `mem::replace(&mut _, mem::zeroed())`.
    ///
    /// ### Why is this bad?
    /// This will lead to undefined behavior even if the
    /// value is overwritten later, because the uninitialized value may be
    /// observed in the case of a panic.
    ///
    /// ### Example
    /// ```no_run
    /// use std::mem;
    ///# fn may_panic(v: Vec<i32>) -> Vec<i32> { v }
    ///
    /// #[allow(deprecated, invalid_value)]
    /// fn myfunc (v: &mut Vec<i32>) {
    ///     let taken_v = unsafe { mem::replace(v, mem::uninitialized()) };
    ///     let new_v = may_panic(taken_v); // undefined behavior on panic
    ///     mem::forget(mem::replace(v, new_v));
    /// }
    /// ```
    ///
    /// The [take_mut](https://docs.rs/take_mut) crate offers a sound solution,
    /// at the cost of either lazily creating a replacement value or aborting
    /// on panic, to ensure that the uninitialized value cannot be observed.
    #[clippy::version = "1.39.0"]
    pub MEM_REPLACE_WITH_UNINIT,
    correctness,
    "`mem::replace(&mut _, mem::uninitialized())` or `mem::replace(&mut _, mem::zeroed())`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `std::mem::replace` on a value of type
    /// `T` with `T::default()`.
    ///
    /// ### Why is this bad?
    /// `std::mem` module already has the method `take` to
    /// take the current value and replace it with the default value of that type.
    ///
    /// ### Example
    /// ```no_run
    /// let mut text = String::from("foo");
    /// let replaced = std::mem::replace(&mut text, String::default());
    /// ```
    /// Is better expressed with:
    /// ```no_run
    /// let mut text = String::from("foo");
    /// let taken = std::mem::take(&mut text);
    /// ```
    #[clippy::version = "1.42.0"]
    pub MEM_REPLACE_WITH_DEFAULT,
    style,
    "replacing a value of type `T` with `T::default()` instead of using `std::mem::take`"
}

impl_lint_pass!(MemReplace =>
    [MEM_REPLACE_OPTION_WITH_NONE, MEM_REPLACE_WITH_UNINIT, MEM_REPLACE_WITH_DEFAULT]);

fn check_replace_option_with_none(cx: &LateContext<'_>, dest: &Expr<'_>, expr_span: Span) {
    // Since this is a late pass (already type-checked),
    // and we already know that the second argument is an
    // `Option`, we do not need to check the first
    // argument's type. All that's left is to get
    // the replacee's expr after peeling off the `&mut`
    let sugg_expr = peel_ref_operators(cx, dest);
    let mut applicability = Applicability::MachineApplicable;
    span_lint_and_sugg(
        cx,
        MEM_REPLACE_OPTION_WITH_NONE,
        expr_span,
        "replacing an `Option` with `None`",
        "consider `Option::take()` instead",
        format!(
            "{}.take()",
            Sugg::hir_with_context(cx, sugg_expr, expr_span.ctxt(), "", &mut applicability).maybe_par()
        ),
        applicability,
    );
}

fn check_replace_with_uninit(cx: &LateContext<'_>, src: &Expr<'_>, dest: &Expr<'_>, expr_span: Span) {
    if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(src.hir_id)
        // check if replacement is mem::MaybeUninit::uninit().assume_init()
        && cx.tcx.is_diagnostic_item(sym::assume_init, method_def_id)
    {
        let Some(top_crate) = std_or_core(cx) else { return };
        let mut applicability = Applicability::MachineApplicable;
        span_lint_and_sugg(
            cx,
            MEM_REPLACE_WITH_UNINIT,
            expr_span,
            "replacing with `mem::MaybeUninit::uninit().assume_init()`",
            "consider using",
            format!(
                "{top_crate}::ptr::read({})",
                snippet_with_applicability(cx, dest.span, "", &mut applicability)
            ),
            applicability,
        );
        return;
    }

    if let ExprKind::Call(repl_func, []) = src.kind
        && let ExprKind::Path(ref repl_func_qpath) = repl_func.kind
        && let Some(repl_def_id) = cx.qpath_res(repl_func_qpath, repl_func.hir_id).opt_def_id()
    {
        if cx.tcx.is_diagnostic_item(sym::mem_uninitialized, repl_def_id) {
            let Some(top_crate) = std_or_core(cx) else { return };
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                MEM_REPLACE_WITH_UNINIT,
                expr_span,
                "replacing with `mem::uninitialized()`",
                "consider using",
                format!(
                    "{top_crate}::ptr::read({})",
                    snippet_with_applicability(cx, dest.span, "", &mut applicability)
                ),
                applicability,
            );
        } else if cx.tcx.is_diagnostic_item(sym::mem_zeroed, repl_def_id)
            && !cx.typeck_results().expr_ty(src).is_primitive()
        {
            span_lint_and_help(
                cx,
                MEM_REPLACE_WITH_UNINIT,
                expr_span,
                "replacing with `mem::zeroed()`",
                None,
                "consider using a default value or the `take_mut` crate instead",
            );
        }
    }
}

fn check_replace_with_default(cx: &LateContext<'_>, src: &Expr<'_>, dest: &Expr<'_>, expr_span: Span) {
    // disable lint for primitives
    let expr_type = cx.typeck_results().expr_ty_adjusted(src);
    if is_non_aggregate_primitive_type(expr_type) {
        return;
    }
    if is_default_equivalent(cx, src) && !in_external_macro(cx.tcx.sess, expr_span) {
        let Some(top_crate) = std_or_core(cx) else { return };
        span_lint_and_then(
            cx,
            MEM_REPLACE_WITH_DEFAULT,
            expr_span,
            format!(
                "replacing a value of type `T` with `T::default()` is better expressed using `{top_crate}::mem::take`"
            ),
            |diag| {
                if !expr_span.from_expansion() {
                    let suggestion = format!("{top_crate}::mem::take({})", snippet(cx, dest.span, ""));

                    diag.span_suggestion(
                        expr_span,
                        "consider using",
                        suggestion,
                        Applicability::MachineApplicable,
                    );
                }
            },
        );
    }
}

pub struct MemReplace {
    msrv: Msrv,
}

impl MemReplace {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: conf.msrv.clone(),
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for MemReplace {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Call(func, [dest, src]) = expr.kind
            // Check that `expr` is a call to `mem::replace()`
            && let ExprKind::Path(ref func_qpath) = func.kind
            && let Some(def_id) = cx.qpath_res(func_qpath, func.hir_id).opt_def_id()
            && cx.tcx.is_diagnostic_item(sym::mem_replace, def_id)
        {
            // Check that second argument is `Option::None`
            if is_res_lang_ctor(cx, path_res(cx, src), OptionNone) {
                check_replace_option_with_none(cx, dest, expr.span);
            } else if self.msrv.meets(msrvs::MEM_TAKE) && is_expr_used_or_unified(cx.tcx, expr) {
                check_replace_with_default(cx, src, dest, expr.span);
            }
            check_replace_with_uninit(cx, src, dest, expr.span);
        }
    }
    extract_msrv_attr!(LateContext);
}
