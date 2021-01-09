use rustc_errors::Applicability;
use rustc_hir::def::{CtorKind, CtorOf, DefKind, Res};
use rustc_hir::{Body, Expr, ExprKind, LangItem, MatchSource, QPath};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::ty::DefIdTree;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::sym;

use crate::utils;
use if_chain::if_chain;

declare_clippy_lint! {
    /// **What it does:**
    /// Suggests alternatives for useless applications of `?` in terminating expressions
    ///
    /// **Why is this bad?** There's no reason to use `?` to short-circuit when execution of the body will end there anyway.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
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
    /// ```rust
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
    pub NEEDLESS_QUESTION_MARK,
    complexity,
    "Suggest `value.inner_option` instead of `Some(value.inner_option?)`. The same goes for `Result<T, E>`."
}

const NEEDLESS_QUESTION_MARK_RESULT_MSRV: RustcVersion = RustcVersion::new(1, 13, 0);
const NEEDLESS_QUESTION_MARK_OPTION_MSRV: RustcVersion = RustcVersion::new(1, 22, 0);

pub struct NeedlessQuestionMark {
    msrv: Option<RustcVersion>,
}

impl NeedlessQuestionMark {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl_lint_pass!(NeedlessQuestionMark => [NEEDLESS_QUESTION_MARK]);

#[derive(Debug)]
enum SomeOkCall<'a> {
    SomeCall(&'a Expr<'a>, &'a Expr<'a>),
    OkCall(&'a Expr<'a>, &'a Expr<'a>),
}

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
        let e = match &expr.kind {
            ExprKind::Ret(Some(e)) => e,
            _ => return,
        };

        if let Some(ok_some_call) = is_some_or_ok_call(self, cx, e) {
            emit_lint(cx, &ok_some_call);
        }
    }

    fn check_body(&mut self, cx: &LateContext<'_>, body: &'_ Body<'_>) {
        // Function / Closure block
        let expr_opt = if let ExprKind::Block(block, _) = &body.value.kind {
            block.expr
        } else {
            // Single line closure
            Some(&body.value)
        };

        if_chain! {
            if let Some(expr) = expr_opt;
            if let Some(ok_some_call) = is_some_or_ok_call(self, cx, expr);
            then {
                emit_lint(cx, &ok_some_call);
            }
        };
    }

    extract_msrv_attr!(LateContext);
}

fn emit_lint(cx: &LateContext<'_>, expr: &SomeOkCall<'_>) {
    let (entire_expr, inner_expr) = match expr {
        SomeOkCall::OkCall(outer, inner) | SomeOkCall::SomeCall(outer, inner) => (outer, inner),
    };

    utils::span_lint_and_sugg(
        cx,
        NEEDLESS_QUESTION_MARK,
        entire_expr.span,
        "Question mark operator is useless here",
        "try",
        format!("{}", utils::snippet(cx, inner_expr.span, r#""...""#)),
        Applicability::MachineApplicable,
    );
}

fn is_some_or_ok_call<'a>(
    nqml: &NeedlessQuestionMark,
    cx: &'a LateContext<'_>,
    expr: &'a Expr<'_>,
) -> Option<SomeOkCall<'a>> {
    if_chain! {
        // Check outer expression matches CALL_IDENT(ARGUMENT) format
        if let ExprKind::Call(path, args) = &expr.kind;
        if let ExprKind::Path(QPath::Resolved(None, path)) = &path.kind;
        if is_some_ctor(cx, path.res) || is_ok_ctor(cx, path.res);

        // Extract inner expression from ARGUMENT
        if let ExprKind::Match(inner_expr_with_q, _, MatchSource::TryDesugar) = &args[0].kind;
        if let ExprKind::Call(called, args) = &inner_expr_with_q.kind;
        if args.len() == 1;

        if let ExprKind::Path(QPath::LangItem(LangItem::TryIntoResult, _)) = &called.kind;
        then {
            // Extract inner expr type from match argument generated by
            // question mark operator
            let inner_expr = &args[0];

            let inner_ty = cx.typeck_results().expr_ty(inner_expr);
            let outer_ty = cx.typeck_results().expr_ty(expr);

            // Check if outer and inner type are Option
            let outer_is_some = utils::is_type_diagnostic_item(cx, outer_ty, sym::option_type);
            let inner_is_some = utils::is_type_diagnostic_item(cx, inner_ty, sym::option_type);

            // Check for Option MSRV
            let meets_option_msrv = utils::meets_msrv(nqml.msrv.as_ref(), &NEEDLESS_QUESTION_MARK_OPTION_MSRV);
            if outer_is_some && inner_is_some && meets_option_msrv {
                return Some(SomeOkCall::SomeCall(expr, inner_expr));
            }

            // Check if outer and inner type are Result
            let outer_is_result = utils::is_type_diagnostic_item(cx, outer_ty, sym::result_type);
            let inner_is_result = utils::is_type_diagnostic_item(cx, inner_ty, sym::result_type);

            // Additional check: if the error type of the Result can be converted
            // via the From trait, then don't match
            let does_not_call_from = !has_implicit_error_from(cx, expr, inner_expr);

            // Must meet Result MSRV
            let meets_result_msrv = utils::meets_msrv(nqml.msrv.as_ref(), &NEEDLESS_QUESTION_MARK_RESULT_MSRV);
            if outer_is_result && inner_is_result && does_not_call_from && meets_result_msrv {
                return Some(SomeOkCall::OkCall(expr, inner_expr));
            }
        }
    }

    None
}

fn has_implicit_error_from(cx: &LateContext<'_>, entire_expr: &Expr<'_>, inner_result_expr: &Expr<'_>) -> bool {
    return cx.typeck_results().expr_ty(entire_expr) != cx.typeck_results().expr_ty(inner_result_expr);
}

fn is_ok_ctor(cx: &LateContext<'_>, res: Res) -> bool {
    if let Some(ok_id) = cx.tcx.lang_items().result_ok_variant() {
        if let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Fn), id) = res {
            if let Some(variant_id) = cx.tcx.parent(id) {
                return variant_id == ok_id;
            }
        }
    }
    false
}

fn is_some_ctor(cx: &LateContext<'_>, res: Res) -> bool {
    if let Some(some_id) = cx.tcx.lang_items().option_some_variant() {
        if let Res::Def(DefKind::Ctor(CtorOf::Variant, CtorKind::Fn), id) = res {
            if let Some(variant_id) = cx.tcx.parent(id) {
                return variant_id == some_id;
            }
        }
    }
    false
}
