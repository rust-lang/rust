use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg};
use clippy_utils::source::{snippet, snippet_with_macro_callsite};
use clippy_utils::sugg::Sugg;
use clippy_utils::ty::{is_type_diagnostic_item, same_type_and_consts};
use clippy_utils::{get_parent_expr, is_trait_method, match_def_path, paths};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, HirId, MatchSource};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `Into`, `TryInto`, `From`, `TryFrom`, or `IntoIter` calls
    /// which uselessly convert to the same type.
    ///
    /// ### Why is this bad?
    /// Redundant code.
    ///
    /// ### Example
    /// ```rust
    /// // format!() returns a `String`
    /// let s: String = format!("hello").into();
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// let s: String = format!("hello");
    /// ```
    #[clippy::version = "1.45.0"]
    pub USELESS_CONVERSION,
    complexity,
    "calls to `Into`, `TryInto`, `From`, `TryFrom`, or `IntoIter` which perform useless conversions to the same type"
}

#[derive(Default)]
pub struct UselessConversion {
    try_desugar_arm: Vec<HirId>,
}

impl_lint_pass!(UselessConversion => [USELESS_CONVERSION]);

#[expect(clippy::too_many_lines)]
impl<'tcx> LateLintPass<'tcx> for UselessConversion {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if e.span.from_expansion() {
            return;
        }

        if Some(&e.hir_id) == self.try_desugar_arm.last() {
            return;
        }

        match e.kind {
            ExprKind::Match(_, arms, MatchSource::TryDesugar) => {
                let e = match arms[0].body.kind {
                    ExprKind::Ret(Some(e)) | ExprKind::Break(_, Some(e)) => e,
                    _ => return,
                };
                if let ExprKind::Call(_, [arg, ..]) = e.kind {
                    self.try_desugar_arm.push(arg.hir_id);
                }
            },

            ExprKind::MethodCall(name, recv, ..) => {
                if is_trait_method(cx, e, sym::Into) && name.ident.as_str() == "into" {
                    let a = cx.typeck_results().expr_ty(e);
                    let b = cx.typeck_results().expr_ty(recv);
                    if same_type_and_consts(a, b) {
                        let sugg = snippet_with_macro_callsite(cx, recv.span, "<expr>").to_string();
                        span_lint_and_sugg(
                            cx,
                            USELESS_CONVERSION,
                            e.span,
                            &format!("useless conversion to the same type: `{}`", b),
                            "consider removing `.into()`",
                            sugg,
                            Applicability::MachineApplicable, // snippet
                        );
                    }
                }
                if is_trait_method(cx, e, sym::IntoIterator) && name.ident.name == sym::into_iter {
                    if let Some(parent_expr) = get_parent_expr(cx, e) {
                        if let ExprKind::MethodCall(parent_name, ..) = parent_expr.kind {
                            if parent_name.ident.name != sym::into_iter {
                                return;
                            }
                        }
                    }
                    let a = cx.typeck_results().expr_ty(e);
                    let b = cx.typeck_results().expr_ty(recv);
                    if same_type_and_consts(a, b) {
                        let sugg = snippet(cx, recv.span, "<expr>").into_owned();
                        span_lint_and_sugg(
                            cx,
                            USELESS_CONVERSION,
                            e.span,
                            &format!("useless conversion to the same type: `{}`", b),
                            "consider removing `.into_iter()`",
                            sugg,
                            Applicability::MachineApplicable, // snippet
                        );
                    }
                }
                if_chain! {
                    if is_trait_method(cx, e, sym::TryInto) && name.ident.name == sym::try_into;
                    let a = cx.typeck_results().expr_ty(e);
                    let b = cx.typeck_results().expr_ty(recv);
                    if is_type_diagnostic_item(cx, a, sym::Result);
                    if let ty::Adt(_, substs) = a.kind();
                    if let Some(a_type) = substs.types().next();
                    if same_type_and_consts(a_type, b);

                    then {
                        span_lint_and_help(
                            cx,
                            USELESS_CONVERSION,
                            e.span,
                            &format!("useless conversion to the same type: `{}`", b),
                            None,
                            "consider removing `.try_into()`",
                        );
                    }
                }
            },

            ExprKind::Call(path, [arg]) => {
                if_chain! {
                    if let ExprKind::Path(ref qpath) = path.kind;
                    if let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id();
                    then {
                        let a = cx.typeck_results().expr_ty(e);
                        let b = cx.typeck_results().expr_ty(arg);
                        if_chain! {
                            if match_def_path(cx, def_id, &paths::TRY_FROM);
                            if is_type_diagnostic_item(cx, a, sym::Result);
                            if let ty::Adt(_, substs) = a.kind();
                            if let Some(a_type) = substs.types().next();
                            if same_type_and_consts(a_type, b);

                            then {
                                let hint = format!("consider removing `{}()`", snippet(cx, path.span, "TryFrom::try_from"));
                                span_lint_and_help(
                                    cx,
                                    USELESS_CONVERSION,
                                    e.span,
                                    &format!("useless conversion to the same type: `{}`", b),
                                    None,
                                    &hint,
                                );
                            }
                        }

                        if_chain! {
                            if match_def_path(cx, def_id, &paths::FROM_FROM);
                            if same_type_and_consts(a, b);

                            then {
                                let sugg = Sugg::hir_with_macro_callsite(cx, arg, "<expr>").maybe_par();
                                let sugg_msg =
                                    format!("consider removing `{}()`", snippet(cx, path.span, "From::from"));
                                span_lint_and_sugg(
                                    cx,
                                    USELESS_CONVERSION,
                                    e.span,
                                    &format!("useless conversion to the same type: `{}`", b),
                                    &sugg_msg,
                                    sugg.to_string(),
                                    Applicability::MachineApplicable, // snippet
                                );
                            }
                        }
                    }
                }
            },

            _ => {},
        }
    }

    fn check_expr_post(&mut self, _: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if Some(&e.hir_id) == self.try_desugar_arm.last() {
            self.try_desugar_arm.pop();
        }
    }
}
