use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg, span_lint_and_then};
use clippy_utils::source::{snippet, snippet_with_applicability};
use clippy_utils::ty::is_non_aggregate_primitive_type;
use clippy_utils::{in_macro, is_default_equivalent, is_lang_ctor, match_def_path, meets_msrv, msrvs, paths};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::LangItem::OptionNone;
use rustc_hir::{BorrowKind, Expr, ExprKind, Mutability, QPath};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Span;
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
    /// ```rust
    /// use std::mem;
    ///
    /// let mut an_option = Some(0);
    /// let replaced = mem::replace(&mut an_option, None);
    /// ```
    /// Is better expressed with:
    /// ```rust
    /// let mut an_option = Some(0);
    /// let taken = an_option.take();
    /// ```
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
    /// ```
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
    /// ```rust
    /// let mut text = String::from("foo");
    /// let replaced = std::mem::replace(&mut text, String::default());
    /// ```
    /// Is better expressed with:
    /// ```rust
    /// let mut text = String::from("foo");
    /// let taken = std::mem::take(&mut text);
    /// ```
    pub MEM_REPLACE_WITH_DEFAULT,
    style,
    "replacing a value of type `T` with `T::default()` instead of using `std::mem::take`"
}

impl_lint_pass!(MemReplace =>
    [MEM_REPLACE_OPTION_WITH_NONE, MEM_REPLACE_WITH_UNINIT, MEM_REPLACE_WITH_DEFAULT]);

fn check_replace_option_with_none(cx: &LateContext<'_>, src: &Expr<'_>, dest: &Expr<'_>, expr_span: Span) {
    if let ExprKind::Path(ref replacement_qpath) = src.kind {
        // Check that second argument is `Option::None`
        if is_lang_ctor(cx, replacement_qpath, OptionNone) {
            // Since this is a late pass (already type-checked),
            // and we already know that the second argument is an
            // `Option`, we do not need to check the first
            // argument's type. All that's left is to get
            // replacee's path.
            let replaced_path = match dest.kind {
                ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, replaced) => {
                    if let ExprKind::Path(QPath::Resolved(None, replaced_path)) = replaced.kind {
                        replaced_path
                    } else {
                        return;
                    }
                },
                ExprKind::Path(QPath::Resolved(None, replaced_path)) => replaced_path,
                _ => return,
            };

            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                MEM_REPLACE_OPTION_WITH_NONE,
                expr_span,
                "replacing an `Option` with `None`",
                "consider `Option::take()` instead",
                format!(
                    "{}.take()",
                    snippet_with_applicability(cx, replaced_path.span, "", &mut applicability)
                ),
                applicability,
            );
        }
    }
}

fn check_replace_with_uninit(cx: &LateContext<'_>, src: &Expr<'_>, dest: &Expr<'_>, expr_span: Span) {
    if_chain! {
        // check if replacement is mem::MaybeUninit::uninit().assume_init()
        if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(src.hir_id);
        if cx.tcx.is_diagnostic_item(sym::assume_init, method_def_id);
        then {
            let mut applicability = Applicability::MachineApplicable;
            span_lint_and_sugg(
                cx,
                MEM_REPLACE_WITH_UNINIT,
                expr_span,
                "replacing with `mem::MaybeUninit::uninit().assume_init()`",
                "consider using",
                format!(
                    "std::ptr::read({})",
                    snippet_with_applicability(cx, dest.span, "", &mut applicability)
                ),
                applicability,
            );
            return;
        }
    }

    if_chain! {
        if let ExprKind::Call(repl_func, repl_args) = src.kind;
        if repl_args.is_empty();
        if let ExprKind::Path(ref repl_func_qpath) = repl_func.kind;
        if let Some(repl_def_id) = cx.qpath_res(repl_func_qpath, repl_func.hir_id).opt_def_id();
        then {
            if cx.tcx.is_diagnostic_item(sym::mem_uninitialized, repl_def_id) {
                let mut applicability = Applicability::MachineApplicable;
                span_lint_and_sugg(
                    cx,
                    MEM_REPLACE_WITH_UNINIT,
                    expr_span,
                    "replacing with `mem::uninitialized()`",
                    "consider using",
                    format!(
                        "std::ptr::read({})",
                        snippet_with_applicability(cx, dest.span, "", &mut applicability)
                    ),
                    applicability,
                );
            } else if cx.tcx.is_diagnostic_item(sym::mem_zeroed, repl_def_id) &&
                    !cx.typeck_results().expr_ty(src).is_primitive() {
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
}

fn check_replace_with_default(cx: &LateContext<'_>, src: &Expr<'_>, dest: &Expr<'_>, expr_span: Span) {
    // disable lint for primitives
    let expr_type = cx.typeck_results().expr_ty_adjusted(src);
    if is_non_aggregate_primitive_type(expr_type) {
        return;
    }
    // disable lint for Option since it is covered in another lint
    if let ExprKind::Path(q) = &src.kind {
        if is_lang_ctor(cx, q, OptionNone) {
            return;
        }
    }
    if is_default_equivalent(cx, src) && !in_external_macro(cx.tcx.sess, expr_span) {
        span_lint_and_then(
            cx,
            MEM_REPLACE_WITH_DEFAULT,
            expr_span,
            "replacing a value of type `T` with `T::default()` is better expressed using `std::mem::take`",
            |diag| {
                if !in_macro(expr_span) {
                    let suggestion = format!("std::mem::take({})", snippet(cx, dest.span, ""));

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
    msrv: Option<RustcVersion>,
}

impl MemReplace {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for MemReplace {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if_chain! {
            // Check that `expr` is a call to `mem::replace()`
            if let ExprKind::Call(func, func_args) = expr.kind;
            if let ExprKind::Path(ref func_qpath) = func.kind;
            if let Some(def_id) = cx.qpath_res(func_qpath, func.hir_id).opt_def_id();
            if match_def_path(cx, def_id, &paths::MEM_REPLACE);
            if let [dest, src] = func_args;
            then {
                check_replace_option_with_none(cx, src, dest, expr.span);
                check_replace_with_uninit(cx, src, dest, expr.span);
                if meets_msrv(self.msrv.as_ref(), &msrvs::MEM_TAKE) {
                    check_replace_with_default(cx, src, dest, expr.span);
                }
            }
        }
    }
    extract_msrv_attr!(LateContext);
}
